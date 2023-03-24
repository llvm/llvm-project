//
// Created by tanmay on 3/23/23.
//

#ifndef LLVM_STATISTICS_H
#define LLVM_STATISTICS_H

struct Statistics {
  double AverageAmplificationFactor;
};

typedef struct Statistics Statistics;

// Constructor for Statistics
Statistics* Statistics_new() {
  Statistics *NewObject;
  if((NewObject = (Statistics *)malloc(sizeof(Statistics))) == NULL) {
    printf("#fAF: Not enough memory for Statistics!");
    exit(EXIT_FAILURE);
  }

  NewObject->AverageAmplificationFactor = 0.0;

  return NewObject;
}

#endif // LLVM_STATISTICS_H
