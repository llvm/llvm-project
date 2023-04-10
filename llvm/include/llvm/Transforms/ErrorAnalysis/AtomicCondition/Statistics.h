//
// Created by tanmay on 3/23/23.
//

#ifndef LLVM_STATISTICS_H
#define LLVM_STATISTICS_H

struct Statistics {
  double HighestAmplificationFactor;
  double HighestConditionNumber;
  double HighestAmplificationDensity;
};

typedef struct Statistics Statistics;

// Constructor for Statistics
Statistics* statisticsNew() {
  Statistics *NewObject;
  if((NewObject = (Statistics *)malloc(sizeof(Statistics))) == NULL) {
    printf("#fAF: Not enough memory for Statistics!");
    exit(EXIT_FAILURE);
  }

  NewObject->HighestAmplificationFactor = 0.0;
  NewObject->HighestConditionNumber = 0.0;
  NewObject->HighestAmplificationDensity = 0.0;

  return NewObject;
}

// Destructor for Statistics
void statisticsDelete(Statistics *Object) {
  free(Object);
}

// Update Highest Amplification Factor
void updateHighestAmplificationFactor(Statistics *Object, double NewValue) {
  if(NewValue > Object->HighestAmplificationFactor) {
    Object->HighestAmplificationFactor = NewValue;
  }
}

// Update Highest Condition Number
void updateHighestConditionNumber(Statistics *Object, double NewValue) {
  if(NewValue > Object->HighestConditionNumber) {
    Object->HighestConditionNumber = NewValue;
  }
}

// Update Highest Amplification Density
void updateHighestAmplificationDensity(Statistics *Object, double NewValue) {
  if(NewValue > Object->HighestAmplificationDensity) {
    Object->HighestAmplificationDensity = NewValue;
  }
}

// Update Statistics
void updateStatistics(Statistics *Object, double NewAmplificationFactor,
                      double NewConditionNumber,
                      double NewAmplificationDensity) {
  updateHighestAmplificationFactor(Object, NewAmplificationFactor);
  updateHighestConditionNumber(Object, NewConditionNumber);
  updateHighestAmplificationDensity(Object, NewAmplificationDensity);
}

// Print Statistics
void printStatistics(Statistics *Object) {
  printf("Highest Amplification Factor: %f\n",
         Object->HighestAmplificationFactor);
  printf("Highest Condition Number: %f\n", Object->HighestConditionNumber);
  printf("Highest Amplification Density: %f\n",
         Object->HighestAmplificationDensity);
}


#endif // LLVM_STATISTICS_H
