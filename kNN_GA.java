import java.lang.Math;
import java.util.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

public class kNN_GA {

  public static int TRAIN_SIZE = 400; // no. training patterns
  public static int VAL_SIZE = 200; // no. validation patterns
  public static int FEATURE_SIZE = 61; // no. of features

  public static double[][] train = new double[TRAIN_SIZE][FEATURE_SIZE]; // data to train
  public static double[][] val = new double[VAL_SIZE][FEATURE_SIZE]; // validation data
  public static int[] train_label = new int[TRAIN_SIZE]; // actual target/class label for train data
  public static int[] val_label = new int[VAL_SIZE]; // actual target/class label for validation data

  public static double accuracy = 0.0;

  public static void main(String[] args) throws IOException {

    System.out.println("Modify the GA function to obtain the best features.");
    System.out.println("Use your optimal k value for kNN that you found for Problem 1");
    System.out.println("DO NOT modify the MAX_POP or MAX_GEN constants or the output file generation");

    Load_Data(TRAIN_SIZE, VAL_SIZE, FEATURE_SIZE); // load data
    GA(); // call GA function

  } // end main loop

  public static void Sort(double[][] sort_array, final int column_sort) {
    Arrays.sort(sort_array, new Comparator<double[]>() {
      @Override
      public int compare(double[] a, double[] b) {
        if (a[column_sort - 1] > b[column_sort - 1])
          return 1;
        else
          return -1;
      }
    });
  }

  public static int Mode(int neigh[]) {
    int modeVal = 0;
    int maxCnt = 0;

    for (int i = 0; i < neigh.length; ++i) {
      int count = 0;
      for (int j = 0; j < neigh.length; ++j) {
        if (neigh[j] == neigh[i])
          count = count + 1;
      }
      if (count > maxCnt) {
        maxCnt = count;
        modeVal = neigh[i];
      }
    }
    return modeVal;
  }

  public static void Load_Data(int TRAIN_SIZE, int VAL_SIZE, int FEATURE_SIZE) throws IOException {

    String train_file = "train_data.txt"; // read training data
    try (Scanner tmp = new Scanner(new File(train_file))) {
      for (int i = 0; i < TRAIN_SIZE; i++)
        for (int j = 0; j < FEATURE_SIZE; j++)
          if (tmp.hasNextDouble())
            train[i][j] = tmp.nextDouble();
      tmp.close();
    }

    String val_file = "val_data.txt"; // read validation data
    try (Scanner tmp = new Scanner(new File(val_file))) {
      for (int i = 0; i < VAL_SIZE; i++)
        for (int j = 0; j < FEATURE_SIZE; j++)
          if (tmp.hasNextDouble())
            val[i][j] = tmp.nextDouble();
      tmp.close();
    }

    String train_label_file = "train_data_label.txt"; // read train label
    try (Scanner tmp = new Scanner(new File(train_label_file))) {
      for (int i = 0; i < TRAIN_SIZE; i++)
        if (tmp.hasNextInt())
          train_label[i] = tmp.nextInt();
      tmp.close();
    }

    String val_label_file = "val_data_label.txt"; // read validation label (to obtain classification accuracy)
    try (Scanner tmp = new Scanner(new File(val_label_file))) {
      for (int i = 0; i < VAL_SIZE; i++)
        if (tmp.hasNextInt())
          val_label[i] = tmp.nextInt();
      tmp.close();
    }

  }

  // Performs a tournament selection by selecting k random individuals within the
  // population, and returning the one with the highest fitness value
  public static boolean[] tourn_select(double[] fitness, boolean[][] sol, int k) {

    int[] chosen_indexes = new int[k];
    double[] chosen_fitnesses = new double[k];

    // Pick the k random individuals
    for (int i = 0; i < k; i++) {
      int index = (int) Math.round(Math.random() * (sol.length - 1));
      chosen_indexes[i] = index;
      chosen_fitnesses[i] = fitness[index];
    }

    // Get the fittest of them all
    int max_index = 0;
    for (int i = 0; i < chosen_indexes.length; i++) {
      if (chosen_fitnesses[i] > chosen_fitnesses[max_index]) {
        max_index = i;
      }
    }

    // for (int i = 0; i < chosen_fitnesses.length; i++) {
    // System.out.println(i + 1 + "Fittnes: " + chosen_fitnesses[i]);
    // }

    // System.out.println("Max: " + fitness[chosen_indexes[max_index]]);
    // System.out.println();
    return sol[chosen_indexes[max_index]];
  }

  public static boolean[] GA() {

    int POP_SIZE = 100; // population size, DO NOT MODIFY
    int MAX_GEN = 50; // maximum generation, DO NOT MODIFY
    boolean[] temp_sol = new boolean[FEATURE_SIZE];
    double[] fitness = new double[POP_SIZE];
    boolean[][] sol = new boolean[POP_SIZE][FEATURE_SIZE];
    boolean[][] new_sol = new boolean[POP_SIZE][FEATURE_SIZE];
    boolean[] final_sol = new boolean[FEATURE_SIZE]; // final best pop

    double featureCountWeighting = 0.0025; // Weighting for (count / feature size), used to put more or less
                                           // emphasis on feature count within the solutions

    // create initial population
    for (int j = 0; j < POP_SIZE; j++) {
      int count = 0;
      for (int k = 0; k < FEATURE_SIZE; k++) {
        sol[j][k] = (Math.random() > 0.5);
        temp_sol[k] = sol[j][k];
        if (temp_sol[k] == true)
          count++;
      }
      // modify fitness to include both increasing accuracy and minimising features
      // Convert to double to ensure non integer division
      fitness[j] = KNN(train, val, train_label, val_label, temp_sol)
          - (((double) count / (double) FEATURE_SIZE) * featureCountWeighting);
      if (count > 40)
        fitness[j] = 0.0;
      // System.out.print(count + " ");
    }
    // System.out.println();

    new_sol = sol; // copy initial array

    for (int gen = 0; gen < MAX_GEN; gen++) { // do for many generations

      System.out.println("GENERATION: " + gen);

      sol = new_sol; // parent copied as children for GA algorithm

      // compute fitness
      for (int j = 0; j < POP_SIZE; j++) {
        int count = 0;
        for (int k = 0; k < FEATURE_SIZE; k++) {
          temp_sol[k] = sol[j][k];
          if (temp_sol[k] == true)
            count++;
        }
        fitness[j] = KNN(train, val, train_label, val_label, temp_sol)
            - (((double) count / (double) FEATURE_SIZE) * featureCountWeighting);
        if (count > 40)
          fitness[j] = 0.0;
        // System.out.print(fitness[j] + " ");
      }

      // #################################################################################################
      // List of top 5 indexes
      int[] maxIndexList = { 0, 0, 0, 0, 0 };

      // Long but CLEAR way of getting top 5 fitnesses from population
      for (int j = 0; j < POP_SIZE; j++) {
        if (fitness[j] > fitness[maxIndexList[0]]) {
          maxIndexList[0] = j;
        } else if (fitness[j] > fitness[maxIndexList[1]]) {
          maxIndexList[1] = j;
        } else if (fitness[j] > fitness[maxIndexList[2]]) {
          maxIndexList[2] = j;
        } else if (fitness[j] > fitness[maxIndexList[3]]) {
          maxIndexList[3] = j;
        } else if (fitness[j] > fitness[maxIndexList[4]]) {
          maxIndexList[4] = j;
        }
      }

      // set top 5 fittest
      boolean[] fittest1 = sol[maxIndexList[0]];
      boolean[] fittest2 = sol[maxIndexList[1]];
      boolean[] fittest3 = sol[maxIndexList[2]];
      boolean[] fittest4 = sol[maxIndexList[3]];
      boolean[] fittest5 = sol[maxIndexList[4]];

      // #################################################################################################
      // #################################################################################################
      // write code to do selection
      boolean tempSel[][] = new boolean[POP_SIZE][FEATURE_SIZE];

      // Uses tournament selection with tourn size 6
      for (int i = 0; i < POP_SIZE; i++) {
        tempSel[i] = tourn_select(fitness, sol, 6);
      }

      sol = tempSel;
      // #################################################################################################
      // #################################################################################################
      // write code to do crossover
      boolean tempCross[][] = new boolean[POP_SIZE][FEATURE_SIZE];

      // POP_SIZE -1 since I replace the current chromosome and the next one
      for (int i = 0; i < POP_SIZE - 1; i++) {
        // 90% chance for crossover to occur
        if (Math.random() <= 0.9) {
          long crossover_point = Math.round((Math.random() * (FEATURE_SIZE - 1)) + 1);
          boolean[] parent1 = sol[i];
          boolean[] parent2 = tourn_select(fitness, sol, 6); // Tourn select parent 2

          boolean[] child1 = new boolean[FEATURE_SIZE];
          boolean[] child2 = new boolean[FEATURE_SIZE];

          for (int j = 0; j < FEATURE_SIZE; j++) {
            if (j < crossover_point) {
              child1[j] = parent1[j];
              child2[j] = parent2[j];
            } else {
              child1[j] = parent2[j];
              child2[j] = parent1[j];
            }
          }

          tempCross[i] = child1;
          tempCross[i + 1] = child2;
        } else {
          tempCross[i] = sol[i];
        }

      }

      sol = tempCross;
      // #################################################################################################
      // #################################################################################################
      // write code to do mutation
      boolean[][] tempMut = new boolean[POP_SIZE][FEATURE_SIZE];

      for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < FEATURE_SIZE; j++) {
          // 1% chance of mutation per gene
          if (Math.random() <= 0.01) {
            tempMut[i][j] = !sol[i][j];
          } else {
            tempMut[i][j] = sol[i][j];
          }
        }
      }

      sol = tempMut;
      // #################################################################################################
      // #################################################################################################
      // code for elitism
      boolean[][] fittestlist = { fittest1, fittest2, fittest3, fittest4, fittest5 };

      // 0 20 40 60 80
      // place the fittest in random places within the array
      // Needed to do it in this complicated way as on occassion, just generating 5
      // random numbers would produce duplicates which would lead to overriding the
      // fittest
      for (int i = 0; i < fittestlist.length; i++) {
        int randInt = (int) Math.round(Math.random() * 19);
        sol[(i * 20) + randInt] = fittestlist[i];
      }

      // #################################################################################################

      int maxAt = 0;
      for (int j = 0; j < POP_SIZE; j++)
        maxAt = fitness[j] > fitness[maxAt] ? j : maxAt;
      for (int k = 0; k < FEATURE_SIZE; k++)
        final_sol[k] = sol[maxAt][k];
      System.out.println("Best fitness = " + fitness[maxAt]);
      // System.out.println("Best fitness index = " + maxAt);

      // update the population
      // copy fitter children as parent for next generation
      new_sol = sol;

    } // end of gen loop

    int channel = 0;
    for (int k = 0; k < FEATURE_SIZE; k++)
      if (final_sol[k]) {
        System.out.print(1 + " ");
        channel++;
      } else
        System.out.print(0 + " ");
    System.out.println("");
    System.out.println("Channel count =  " + channel);

    System.out.println("Accuracy =  " +

        KNN(train, val, train_label, val_label, final_sol));

    // write best solution (features) to file, DO NOT MODIFY
    try {
      PrintWriter writer = new PrintWriter("kNN_GA_output.txt", "UTF-8");
      for (int j = 0; j < FEATURE_SIZE; j++)
        if (final_sol[j])
          writer.print("1 ");
        else
          writer.print("0 ");
      writer.close();
    } catch (Exception e) {
      System.out.println(e);
    }

    return final_sol;

  }

  public static double KNN(double[][] train, double[][] val, int[] train_label, int[] val_label, boolean[] sol) {
    double[][] dist_label = new double[TRAIN_SIZE][2]; // distance array, no of columns+1 to accomodate distance
    double[] y = new double[FEATURE_SIZE];
    double[] x = new double[FEATURE_SIZE];

    int num_neighbour = 22; // optimal k value

    int[] neighbour = new int[num_neighbour];
    int[] predicted_class = new int[VAL_SIZE];

    for (int j = 0; j < VAL_SIZE; j++) {// for every validation data
      for (int f = 0; f < FEATURE_SIZE; f++)
        if (sol[f])
          y[f] = val[j][f];
        else
          y[f] = 0.0;

      for (int i = 0; i < TRAIN_SIZE; i++) {
        for (int f = 0; f < FEATURE_SIZE; f++)
          if (sol[f])
            x[f] = train[i][f];
          else
            x[f] = 0.0;

        double sum = 0.0; // Euclidean distance
        for (int f = 0; f < FEATURE_SIZE; f++)
          sum = sum + ((x[f] - y[f]) * (x[f] - y[f]));

        dist_label[i][0] = Math.sqrt(sum);
        dist_label[i][1] = train_label[i];
      }

      Sort(dist_label, 1); // Sorting distance

      for (int n = 0; n < num_neighbour; n++) // training label from required neighbours
        neighbour[n] = (int) dist_label[n][1];

      predicted_class[j] = Mode(neighbour);

    } // end val data loop

    int success = 0;
    for (int j = 0; j < VAL_SIZE; j++)
      if (predicted_class[j] == val_label[j])
        success++;
    accuracy = (success * 1.0) / VAL_SIZE;
    // System.out.print(accuracy + " ");

    return accuracy;
  }

} // end class loop