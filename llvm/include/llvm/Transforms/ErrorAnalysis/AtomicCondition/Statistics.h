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
Statistics* fSTStatisticsNew() {
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
void fSTStatisticsDelete(Statistics *Object) {
  free(Object);
}

// Update Highest Amplification Factor
void fSTUpdateHighestAmplificationFactor(Statistics *Object, double NewValue) {
  if(NewValue > Object->HighestAmplificationFactor) {
    Object->HighestAmplificationFactor = NewValue;
  }
}

// Update Highest Condition Number
void fSTUpdateHighestConditionNumber(Statistics *Object, double NewValue) {
  if(NewValue > Object->HighestConditionNumber) {
    Object->HighestConditionNumber = NewValue;
  }
}

// Update Highest Amplification Density
void fSTUpdateHighestAmplificationDensity(Statistics *Object, double NewValue) {
  if(NewValue > Object->HighestAmplificationDensity) {
    Object->HighestAmplificationDensity = NewValue;
  }
}

// Update Statistics
void fSTUpdateStatistics(Statistics *Object, double NewAmplificationFactor,
                      double NewConditionNumber,
                      double NewAmplificationDensity) {
  fSTUpdateHighestAmplificationFactor(Object, NewAmplificationFactor);
  fSTUpdateHighestConditionNumber(Object, NewConditionNumber);
  fSTUpdateHighestAmplificationDensity(Object, NewAmplificationDensity);
}

// Print Statistics
void fSTPrintStatistics(Statistics *Object) {
  printf("Highest Amplification Factor: %0.15lf\n",
         Object->HighestAmplificationFactor);
  printf("Highest Condition Number: %0.15lf\n", Object->HighestConditionNumber);
  printf("Highest Amplification Density: %0.15lf\n",
         Object->HighestAmplificationDensity);
}


#endif // LLVM_STATISTICS_H
