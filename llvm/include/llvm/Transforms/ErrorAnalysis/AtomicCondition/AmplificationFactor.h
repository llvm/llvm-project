//
// Created by tanmay on 7/11/22.
//

#ifndef LLVM_AMPLIFICATIONFACTOR_H
#define LLVM_AMPLIFICATIONFACTOR_H

#include "AtomicCondition.h"
#include "../Utilities/RunTimeUtilities.h"
#include "Statistics.h"
#include "stdlib.h"

/*----------------------------------------------------------------------------*/
/* Constants                                                                  */
/*----------------------------------------------------------------------------*/
#define AF_ITEM_LIST_SIZE 10000

// ULP error introduced by each operation on x86_64 architecture as given in
// https://www.gnu.org/software/libc/manual/html_node/Errors-in-Math-Functions.html
float fp32OpError[] = {
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    2e-6,
    2e-6,
    2e-6,
    1e-6,
    1e-6,
    1e-6, // Does not have a known error
    1e-6, // Truncing is not a GNU library function. Giving it 1 ULP error
    1e-6, // Extending is not a GNU library function. Giving it 1 ULP error
    1e-6,
};

double fp64OpError[] = {
    1e-16,
    1e-16,
    1e-16,
    1e-16,
    1e-16,
    1e-16,
    1e-16, // Does not have a known error
    1e-16,
    1e-16,
    1e-16,
    2e-16,
    2e-16,
    2e-16,
    1e-16,
    1e-16,
    1e-16, // Does not have a known error
    1e-16, // Truncing is not a GNU library function. Giving it 1 ULP error
    1e-16, // Extending is not a GNU library function. Giving it 1 ULP error
    1e-16,
};

/*----------------------------------------------------------------------------*/
/* Data Structures, Associated Functions and Types                            */
/*----------------------------------------------------------------------------*/

/* ------------------------------- AFProduct -------------------------------*/

struct AFProduct {
  int ItemId; // Unique identifier for an instance.

  // List of Pointers to Amplification data corresponding to executions with separate inputs.
  // (We assume the computation paths do not change for separate inputs.)
  ACItem **Factors; // Pointers to ACItems of the root operations.
  int *ACWRTs;  // Operand WRT which the AC we consider in this product from ACItem.
  struct AFProduct **ProductTails; // Pointers to AFProducts of the next operations along a computation path.
  char **Inputs; // Register names of the inputs.
  double *AFs; // Amplification Factors.

  // Metadata
  int Height; // Height of the Node in the Computation DAG.
  int NumberOfInputs; // Number of inputs to the Node.
};

typedef struct AFProduct AFProduct;

void fAFCreateAFProduct(AFProduct **AddressToAllocateAt,
                          int ItemId,
                          int LengthOfLists) {
  if((*AddressToAllocateAt = (AFProduct *)malloc(sizeof(AFProduct))) == NULL) {
    printf("#fAF: Not enough memory for AFProduct!");
    exit(EXIT_FAILURE);
  }

  (*AddressToAllocateAt)->ItemId = ItemId;

  if(((*AddressToAllocateAt)->Factors = (ACItem **)malloc(sizeof(ACItem *) *
                                                           LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for Factors!");
    exit(EXIT_FAILURE);
  }
  // Initialize all Factors to NULL
  for (int I = 0; I < LengthOfLists; ++I) {
    (*AddressToAllocateAt)->Factors[I] = NULL;
  }

  if(((*AddressToAllocateAt)->ACWRTs = (int *)malloc(sizeof(int *) *
                                                      LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for AFs!");
    exit(EXIT_FAILURE);
  }
  // Initialize all ACWRTs to -1 which denotes that the ACWRT is not set.
  for (int I = 0; I < LengthOfLists; ++I) {
    (*AddressToAllocateAt)->ACWRTs[I] = -1;
  }

  if(((*AddressToAllocateAt)->ProductTails = (AFProduct **)malloc(sizeof(AFProduct *) *
                                                           LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for ProductTails!");
    exit(EXIT_FAILURE);
  }
  // Initialize all ProductTails to NULL
  for (int I = 0; I < LengthOfLists; ++I) {
    (*AddressToAllocateAt)->ProductTails[I] = NULL;
  }

  if(((*AddressToAllocateAt)->Inputs = (char **)malloc(sizeof(char *) *
                                                                   LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for Inputs!");
    exit(EXIT_FAILURE);
  }
  // Initialize all Inputs to NULL
  for (int I = 0; I < LengthOfLists; ++I) {
    (*AddressToAllocateAt)->Inputs[I] = NULL;
  }

  if(((*AddressToAllocateAt)->AFs = (double *)malloc(sizeof(double) *
                                                        LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for AFs!");
    exit(EXIT_FAILURE);
  }
  // Initialize all AFs to 1.0 (i.e. no amplification)
  for (int I = 0; I < LengthOfLists; ++I) {
    (*AddressToAllocateAt)->AFs[I] = 1.0;
  }

  (*AddressToAllocateAt)->Height = 1;
  (*AddressToAllocateAt)->NumberOfInputs = 0;

  return ;
}

void fAFPrintAFProduct(AFProduct *ProductObject) {
  printf("\t\t\t\"ProductItemId\": %d,\n", ProductObject->ItemId);

  printf("\t\t\t\"Height\": %d,\n",
         ProductObject->Height);

  printf("\t\t\t\"Number of Inputs\": %d,\n",
         ProductObject->NumberOfInputs);
  printf("\t\t\t\"ACItemIds\": [%d", ProductObject->Factors[0]->ItemId);
  for (int I = 1; I < ProductObject->NumberOfInputs; ++I) {
    printf(",%d", ProductObject->Factors[I]->ItemId);
  }
  printf("],\n");

  printf("\t\t\t\"ACWRTs\": [%d", ProductObject->ACWRTs[0]);
  for (int I = 1; I < ProductObject->NumberOfInputs; ++I) {
    printf(",%d", ProductObject->ACWRTs[I]);
  }
  printf("],\n");

  printf("\t\t\t\"ProductTailItemIds\": [%d",
         ProductObject->ProductTails[0] != NULL
             ? ProductObject->ProductTails[0]->ItemId
             : -1);
  for (int I = 1; I < ProductObject->NumberOfInputs; ++I) {
    printf(",%d", ProductObject->ProductTails[I] != NULL
                      ? ProductObject->ProductTails[I]->ItemId
                      : -1);
  }
  printf("],\n");

  printf("\t\t\t\"Inputs\": [\"%s\"", ProductObject->Inputs[0]);
  for (int I = 1; I < ProductObject->NumberOfInputs; ++I) {
    printf(",\"%s\"", ProductObject->Inputs[I]);
  }
  printf("],\n");


  if(isinf(ProductObject->AFs[0]))
    printf("\t\t\t\"AFs\": [ \"inf\"");
  else
    printf("\t\t\t\"AFs\": [%0.15lf", ProductObject->AFs[0]);
  for (int I = 1; I < ProductObject->NumberOfInputs; ++I) {
    if(isinf(ProductObject->AFs[I]))
      printf(", \"inf\"");
    else
      printf(", %0.15lf", ProductObject->AFs[I]);
  }
  printf("],\n");
}

AFProduct **fAFFlattenAFComponentsPath(AFProduct *ProductObject) {
  AFProduct *ProductObjectWalker = ProductObject;
  AFProduct **ProductPath;

  if((ProductPath =
           (AFProduct **)malloc(sizeof(AFProduct*) * ProductObject->Height)) == NULL) {
    printf("#fAF: Not enough memory for AFProduct pointers!");
    exit(EXIT_FAILURE);
  }
  ProductPath[0] = &*ProductObjectWalker;
//  fAFPrintAFProduct(ProductObjectWalker);
//  printf("\n");

  for (int I = 1; I < ProductObject->Height; ++I) {
    ProductPath[I] = &*(ProductObjectWalker->ProductTails[0]);
    ProductObjectWalker = ProductObjectWalker->ProductTails[0];
//    fAFPrintAFProduct(ProductObjectWalker);
//    printf("\n");
  }

  return ProductPath;
}

// Function to write NumObjects number of AFProducts from ObjectPointerList into
// file with descriptor FP.
void fAFStoreAFProducts(FILE *FP, AFProduct **ObjectPointerList, uint64_t NumObjects) {
  assert(FP != NULL && "File descriptor is NULL.");
  assert(ObjectPointerList != NULL && "ObjectPointerList is NULL");

  fprintf(FP, "{\n");

  fprintf(FP, "\t\"AFs\": [\n");
  for (uint64_t J = 0; J < NumObjects; ++J) {
    AFProduct **ProductPath= fAFFlattenAFComponentsPath(ObjectPointerList[J]);
    fprintf(FP,
            "\t\t{\n"
            "\t\t\t\"ProductItemId\": %d,\n",
            ObjectPointerList[J]->ItemId);

//    int M = findMaxIndex(ObjectPointerList[J]->AFs, ObjectPointerList[J]->NumberOfInputs);

    fprintf(FP, "\t\t\t\"ACItemIds\": [%d", ObjectPointerList[J]->Factors[0]->ItemId);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",%d", ObjectPointerList[J]->Factors[I]->ItemId);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"ACWRTs\": [%d", ObjectPointerList[J]->ACWRTs[0]);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",%d", ObjectPointerList[J]->ACWRTs[I]);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"ProductTailItemIds\": [%d", ObjectPointerList[J]->
                                                             ProductTails[0]!=NULL?ObjectPointerList[J]->ProductTails[0]->ItemId:
                                                         -1);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",%d", ObjectPointerList[J]->
                                 ProductTails[I]!=NULL?ObjectPointerList[J]->ProductTails[I]->ItemId:
                             -1);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"Inputs\": [\"%s\"", ObjectPointerList[J]->Inputs[0]);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",\"%s\"", ObjectPointerList[J]->Inputs[I]);
    }
    fprintf(FP, "],\n");

    if(isinf(ObjectPointerList[J]->AFs[0]))
      fprintf(FP,"\t\t\t\"AFs\": [ \"inf\"");
    else
      fprintf(FP,"\t\t\t\"AFs\": [%0.15lf", ObjectPointerList[J]->AFs[0]);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
      if(isinf(ObjectPointerList[J]->AFs[I]))
        fprintf(FP, ", \"inf\"");
      else
        fprintf(FP, ",%0.15lf", ObjectPointerList[J]->AFs[I]);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"Path(AFProductIds)\": [%d", ProductPath[0]->ItemId);
    for (int K = 1; K < ObjectPointerList[J]->Height; ++K)
      fprintf(FP, ", %d", ProductPath[K]->ItemId);

    if ((uint64_t)J == NumObjects-1)
      fprintf(FP, "]\n\t\t}\n");
    else
      fprintf(FP, "]\n\t\t},\n");
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");
}

/* ---------------------------------- AFItem ---------------------------------*/

struct AFItem {
  int ItemId; // Unique identifier for an instance of AFItem.
  int FunctionEvaluationId; // Identifier for the function call in which the instruction instance
                            // about which this AFItem provides information about lies.
  int InstructionId; // Identifier for the instruction about which this AFItem informs.
  int InstructionExecutionId; // Identifier for the instance of the instruction about which this AFItem informs.

  AFProduct** Components; // List of AFProducts corresponding the roots of the
                          // computation paths starting at the instruction which
                          // this AFItem informs about.
  int NumAFComponents; // Number of AFProducts in the Components list.
  int RootNode; // 1 if this AFItem belongs to a root node of a computation DAG, 0 otherwise.
};

typedef struct AFItem AFItem;

void fAFCreateAFItem(AFItem **AddressToAllocateAt,
                     int ItemId,
                     int FunctionEvaluationId,
                     int InstructionId,
                     int InstructionExecutionId,
                     int NumAFComponents) {
  if((*AddressToAllocateAt = (AFItem *)malloc(sizeof(AFItem))) == NULL) {
    printf("#fAF: Not enough memory for AFItem!");
    exit(EXIT_FAILURE);
  }

  (*AddressToAllocateAt)->ItemId = ItemId;
  (*AddressToAllocateAt)->FunctionEvaluationId = FunctionEvaluationId;
  (*AddressToAllocateAt)->InstructionId = InstructionId;
  (*AddressToAllocateAt)->InstructionExecutionId = InstructionExecutionId;
  (*AddressToAllocateAt)->Components = NULL;
  (*AddressToAllocateAt)->NumAFComponents = NumAFComponents;
  // By default, all newly created AFItems are root nodes.
  (*AddressToAllocateAt)->RootNode = 1;

  return ;
}

// Gets the index of the AFItem * with InstructionId and InstructionExecutionId
// from List of AFItem*.
int fAFGetAFItemFromList(AFItem **AFItemList,
                             int NumAFItems,
                             int InstructionId,
                             int InstructionExecutionId) {
  for (int I = 0; I < NumAFItems; ++I)
    if (AFItemList[I]->InstructionId == InstructionId &&
        AFItemList[I]->InstructionExecutionId == InstructionExecutionId)
      return I;

  return -1;
}

// Function to write NumObjects number of AFItems from ObjectPointerList into
// file with descriptor FP.
void fAFStoreAFItems(FILE *FP, AFItem **ObjectPointerList, uint64_t NumObjects) {
  assert(FP != NULL && "File descriptor is NULL.");
  assert(ObjectPointerList != NULL && "ObjectPointerList is NULL");

  fprintf(FP, "{\n");

  fprintf(FP, "\t\"AFs\": [\n");
  int K = 0;
  while ((uint64_t)K < NumObjects) {
//    printf("K: %d\n", K);
    for (int J = 0; J < ObjectPointerList[K]->NumAFComponents; ++J) {
      AFProduct **ProductPath= fAFFlattenAFComponentsPath(ObjectPointerList[K]->Components[J]);
//      printf("\t ProductItemId: %d,\n",
//              ObjectPointerList[K]->Components[J]->ItemId);
      fprintf(FP,
              "\t\t{\n"
              "\t\t\t\"ProductItemId\": %d,\n",
              ObjectPointerList[K]->Components[J]->ItemId);

      fprintf(FP,
              "\t\t\t\"Height\": %d,\n",
              ObjectPointerList[K]->Components[J]->Height);

      fprintf(FP, "\t\t\t\"ACItemIds\": [%d", ObjectPointerList[K]->Components[J]->Factors[0]->ItemId);
      for (int I = 1; I < ObjectPointerList[K]->Components[J]->NumberOfInputs; ++I) {
        fprintf(FP, ",%d", ObjectPointerList[K]->Components[J]->Factors[I]->ItemId);
      }
      fprintf(FP, "],\n");

      fprintf(FP, "\t\t\t\"ACWRTs\": [%d", ObjectPointerList[K]->Components[J]->ACWRTs[0]);
      for (int I = 1; I < ObjectPointerList[K]->Components[J]->NumberOfInputs; ++I) {
        fprintf(FP, ",%d", ObjectPointerList[K]->Components[J]->ACWRTs[I]);
      }
      fprintf(FP, "],\n");

      fprintf(FP, "\t\t\t\"ProductTailItemIds\": [%d", ObjectPointerList[K]->Components[J]->
                                                               ProductTails[0]!=NULL?ObjectPointerList[K]->Components[J]->ProductTails[0]->ItemId:
                                                           -1);
      for (int I = 1; I < ObjectPointerList[K]->Components[J]->NumberOfInputs; ++I) {
        fprintf(FP, ",%d", ObjectPointerList[K]->Components[J]->
                                   ProductTails[I]!=NULL?ObjectPointerList[K]->Components[J]->ProductTails[I]->ItemId:
                               -1);
      }
      fprintf(FP, "],\n");

      fprintf(FP, "\t\t\t\"Inputs\": [\"%s\"", ObjectPointerList[K]->Components[J]->Inputs[0]);
      for (int I = 1; I < ObjectPointerList[K]->Components[J]->NumberOfInputs; ++I) {
        fprintf(FP, ",\"%s\"", ObjectPointerList[K]->Components[J]->Inputs[I]);
      }
      fprintf(FP, "],\n");

      if(isinf(ObjectPointerList[K]->Components[J]->AFs[0]))
        fprintf(FP,"\t\t\t\"AFs\": [ \"inf\"");
      else
        fprintf(FP,"\t\t\t\"AFs\": [%0.15lf", ObjectPointerList[K]->Components[J]->AFs[0]);
      for (int I = 1; I < ObjectPointerList[K]->Components[J]->NumberOfInputs; ++I) {
        if(isinf(ObjectPointerList[K]->Components[J]->AFs[I]))
          fprintf(FP, ", \"inf\"");
        else
          fprintf(FP, ",%0.15lf", ObjectPointerList[K]->Components[J]->AFs[I]);
      }
      fprintf(FP, "],\n");

      fprintf(FP, "\t\t\t\"Path(AFProductIds)\": [%d", ProductPath[0]->ItemId);

      for(int I = 1; I < ObjectPointerList[K]->Components[J]->Height; ++I)
        fprintf(FP, ", %d", ProductPath[I]->ItemId);

      if ((uint64_t)K == NumObjects-1 && J == ObjectPointerList[K]->NumAFComponents-1)
        fprintf(FP, "]\n\t\t}\n");
      else
        fprintf(FP, "]\n\t\t},\n");
    }

    K++;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");
}
/* --------------------------------- AFTable ---------------------------------*/

struct AFTable {
  uint64_t ListLength;
  struct AFItem **AFItems;
};

typedef struct AFTable AFTable;

void fAFCreateAFTable(AFTable **AddressToAllocateAt) {
#if FAF_DEBUG
  printf("Initializing Amplification Factor Module\n");
#endif

  if((*AddressToAllocateAt = (AFTable *)malloc(sizeof(AFTable))) == NULL) {
    printf("#fAF: Not enough memory for AFItem!");
    exit(EXIT_FAILURE);
  }

#if FAF_DEBUG
  printf("Amplification Factor Module Initialized\n");
#endif
}

// Gets the number of AFItems in the AFTable that are RootNodes (These will correspond
// to the number of computation DAGs in the program)
int getNumberOfComputationDAGs(AFTable *AFTable) {
  int NumberOfComputationDAGs = 0;
  for (uint64_t I = 0; I < AFTable->ListLength; ++I) {
    if (AFTable->AFItems[I]->RootNode)
      NumberOfComputationDAGs++;
  }
  return NumberOfComputationDAGs;
}

// Gets the number of computation paths that start from some RootNode.
int getNumberOfComputationPaths(AFTable *AFTable) {
  int NumberOfComputationPaths = 0;
  for (uint64_t I = 0; I < AFTable->ListLength; ++I) {
    if (AFTable->AFItems[I]->RootNode)
      NumberOfComputationPaths += AFTable->AFItems[I]->NumAFComponents;
  }
  return NumberOfComputationPaths;
}

// Compute the Average Amplification Factor incrementally.
double getAverageAmplificationFactor(AFTable *AFTable, int NumFunctionEvaluations) {
  double* AverageAmplificationFactor;

  if((AverageAmplificationFactor = (double *)malloc(sizeof(double)*NumFunctionEvaluations)) == NULL) {
    printf("#fAF: Not enough memory for AverageAmplificationFactors!");
    exit(EXIT_FAILURE);
  }
  // Initialize the AverageAmplificationFactor array to 0.
  for (int I = 0; I < NumFunctionEvaluations; ++I) {
    AverageAmplificationFactor[I] = 0;
  }

  int NumberOfComputationPaths = 0;

  for (uint64_t I = 0; I < AFTable->ListLength; ++I) {
    if (AFTable->AFItems[I]->RootNode) {
      for (int J = 0; J < AFTable->AFItems[I]->NumAFComponents; ++J, ++NumberOfComputationPaths) {
        for (int K = 0; K < NumFunctionEvaluations; ++K) {
          // Average is computed as m_n = m_(n-1) + (a_n - m_(n-1))/n
          AverageAmplificationFactor[K] +=
              (AFTable->AFItems[I]->Components[J]->AFs[K] -
               AverageAmplificationFactor[K]) /
              (NumberOfComputationPaths + 1);
        }
      }
    }
  }

  // Find the greatest from AverageAmplificationFactor array.
  double GreatestAverageAmplificationFactor = 0;
  for (int I = 0; I < NumFunctionEvaluations; ++I) {
    if (AverageAmplificationFactor[I] > GreatestAverageAmplificationFactor)
      GreatestAverageAmplificationFactor = AverageAmplificationFactor[I];
  }

  assert(NumberOfComputationPaths == getNumberOfComputationPaths(AFTable));
  return GreatestAverageAmplificationFactor;
}

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/

int AFItemCounter;
int AFComponentCounter;
int PlotDataCounter;
int InstructionExecutionCounter;
int FunctionInstanceCounter;
AFTable *AFs;
AFProduct **Paths;
Statistics *Stats;

/*----------------------------------------------------------------------------*/
/* Utility Functions                                                          */
/*----------------------------------------------------------------------------*/

#pragma clang optimize off
void fAFfp32markForResult(float Res) {
  return ;
}

void fAFfp64markForResult(double Res) {
  return ;
}

#pragma clang optimize on

int min(int A, int B) {
  return (A > B)? B : A;
}

AFProduct **fAFFlattenAllComponentPaths() {
  if((Paths =
           (AFProduct **)malloc(sizeof(AFProduct*) * AFComponentCounter)) == NULL) {
    printf("#fAF: Not enough memory for AFProduct pointers!");
    exit(EXIT_FAILURE);
  }

  for (uint64_t I = 0; I < AFs->ListLength; ++I)
    for (int J = 0; J < AFs->AFItems[I]->NumAFComponents; ++J)
      Paths[AFs->AFItems[I]->Components[J]->ItemId] = &*(AFs->AFItems[I]->Components[J]);

  return Paths;
}

void fAFStoreInFile(AFItem **ObjectToStore) {
//  printf("\nCollecting Data for Plot\n");
  char File[5000];
  // Create a directory if not present
  const int DirectoryNameLen = sizeof LOG_DIRECTORY_NAME;
  char DirectoryName[DirectoryNameLen];
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  File[0] = '\0';
  strcat(
      strcpy(
          File,
          DirectoryName),
      "/PlotData.json");

  char ExecutionId[5000];
  fACGenerateExecutionID(ExecutionId);

  // Table Output
  FILE *FP = fopen(File, "a");

  fprintf(FP, "\t\"%s_%d\": [\n", ExecutionId, PlotDataCounter);

  for (int J = 0; J < (*ObjectToStore)->NumAFComponents; ++J) {
    AFProduct **ProductPath= fAFFlattenAFComponentsPath((*ObjectToStore)->Components[J]);
    fprintf(FP,
            "\t\t{\n"
            "\t\t\t\"ProductItemId\": %d,\n",
            (*ObjectToStore)->Components[J]->ItemId);

    fprintf(FP, "\t\t\t\"ACItemIds\": [%d", (*ObjectToStore)->Components[J]->Factors[0]->ItemId);
    for (int I = 1; I < (*ObjectToStore)->Components[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",%d", (*ObjectToStore)->Components[J]->Factors[I]->ItemId);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"ACItemStrings\": [%d", (*ObjectToStore)->Components[J]->ACWRTs[0]);
    for (int I = 1; I < (*ObjectToStore)->Components[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",%d", (*ObjectToStore)->Components[J]->ACWRTs[I]);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"ProductTailItemIds\": [%d", (*ObjectToStore)->Components[J]->
                                                             ProductTails[0]!=NULL?(*ObjectToStore)->Components[J]->ProductTails[0]->ItemId:
                                                         -1);
    for (int I = 1; I < (*ObjectToStore)->Components[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",%d", (*ObjectToStore)->Components[J]->
                                 ProductTails[I]!=NULL?(*ObjectToStore)->Components[J]->ProductTails[I]->ItemId:
                             -1);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"Inputs\": [\"%s\"", (*ObjectToStore)->Components[J]->Inputs[0]);
    for (int I = 1; I < (*ObjectToStore)->Components[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",\"%s\"", (*ObjectToStore)->Components[J]->Inputs[I]);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"AFs\": [%0.15lf", (*ObjectToStore)->Components[J]->AFs[0]);
    for (int I = 1; I < (*ObjectToStore)->Components[J]->NumberOfInputs; ++I) {
      if(isinf((*ObjectToStore)->Components[J]->AFs[I]))
        fprintf(FP, ", \"inf\"");
      else
        fprintf(FP, ",%0.15lf", (*ObjectToStore)->Components[J]->AFs[I]);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"Path(AFProductIds)\": [%d", ProductPath[0]->ItemId);
    for (int K = 1; K < (*ObjectToStore)->Components[J]->Height; ++K)
      fprintf(FP, ", %d", ProductPath[K]->ItemId);

    if (J == (*ObjectToStore)->NumAFComponents-1)
      fprintf(FP, "]\n\t\t}\n");
    else
      fprintf(FP, "]\n\t\t},\n");
  }

  fprintf(FP, "\t],\n");

  fclose(FP);

  PlotDataCounter++;
  return ;
}

int fAFisMemoryOpInstruction(char *InstructionString) {
  if(strstr(InstructionString, "load")!=NULL ||
      strstr(InstructionString, "alloca")!=NULL)
    return 1;
  return 0;
}

//int fAFComparator(const void *A, const void *B) {
//  double AF1 = fabs((*(AFProduct **)A)->AFs[0]);
//  double AF2 = fabs((*(AFProduct **)B)->AFs[0]);
//
//  if(AF2 > AF1)
//    return 1;
//  if(AF2 == AF1)
//    return 0;
//  return -1;
//}

int fAFComparator(const void *A, const void *B) {
  double AF1 = fabs((*(AFProduct **)A)->AFs[0]);
  for (int I = 0; I < (*(AFProduct **)A)->NumberOfInputs; ++I) {
    if(fabs((*(AFProduct **)A)->AFs[I]) > AF1)
      AF1 = fabs((*(AFProduct **)A)->AFs[I]);
  }
  double AF2 = fabs((*(AFProduct **)B)->AFs[0]);
  for (int I = 0; I < (*(AFProduct **)B)->NumberOfInputs; ++I) {
    if(fabs((*(AFProduct **)B)->AFs[I]) > AF2)
      AF2 = fabs((*(AFProduct **)B)->AFs[I]);
  }

  if(AF2 > AF1)
    return 1;
  if(AF2 == AF1)
    return 0;
  return -1;
}

/*----------------------------------------------------------------------------*/
/* Memory Allocators                                                          */
/*----------------------------------------------------------------------------*/

void fAFInitialize() {
  fACCreate();
  AFItemCounter = 0;
  AFComponentCounter = 0;
  PlotDataCounter = 0;

  // Allocating Memory to the AF table itself
  fAFCreateAFTable(&AFs);

  // Allocating Memory for the AFItem Pointers
  if((AFs->AFItems =
           (AFItem **)malloc(sizeof(AFItem*) * AF_ITEM_LIST_SIZE)) == NULL) {
    printf("#fAF: Not enough memory for AFItem pointers!");
    exit(EXIT_FAILURE);
  }
  AFs->ListLength = 0;

  // Allocating Memory for Stats
  Stats = statisticsNew();

  return ;
}

/*----------------------------------------------------------------------------*/
/* Analysis Functions                                                         */
/*----------------------------------------------------------------------------*/

// Inputs: AC Record corresponding to the current instruction,
// AF Records corresponding the operands, Number of Operands, Instruction Id,
// Number of times the instruction is to be executed.
// Note: The current instruction information can be found in the AC record.
// Return the AFItem
#if MEMORY_OPT
AFItem **fAFComputeAF(ACItem **AC, AFItem ***AFItemWRTOperands,
                      int NumOperands,
                      int FunctionEvaluationId,
                      int InstructionId,
                      int NumFunctionEvaluations) {
#if FAF_DEBUG >= 2
  printf("\n\tCreating AFItem\n");
  printf("\t\tACId: %d\n", (*AC)->ItemId);
  printf("\t\tFunctionEvaluationId: %d\n", FunctionEvaluationId);
  printf("\t\tInstructionId: %d\n", InstructionId);
  printf("\t\tInstructionExecutionId: %d\n", InstructionExecutionCounter);
  printf("\t\tNumber of Operands: %d\n", NumOperands);
  printf("\t\tRootNode: %s\n", (*AC)->ResultVar);
  printf("\t\tComponents:\n");
  for (int OperandIndex = 0; OperandIndex < NumOperands; ++OperandIndex) {
    if (AFItemWRTOperands[OperandIndex] != NULL)
      printf("\t\t\tAFId %d #Components: %d\n",
             (*AFItemWRTOperands[OperandIndex])->ItemId,
             (*AFItemWRTOperands[OperandIndex])->NumAFComponents);
    else
      printf("\t\t\tAFId %d #Components: 0\n", -1);
  }

  printf("\n");
#endif

  //  Create a new AF record and initialize data members and allocate memory for
  //  AFComponents array.
  AFItem *UpdatingAFItem = NULL;

  fAFCreateAFItem(&UpdatingAFItem, AFItemCounter, FunctionEvaluationId,
                  InstructionId,InstructionExecutionCounter, 0);
  AFItemCounter++;

  int TotalAFComponents = 0;
  TotalAFComponents += NumOperands;

#if FAF_DEBUG >= 3
  printf("\t\tTotalAFComponents: %d\n\n", TotalAFComponents);
#endif

  if ((UpdatingAFItem->Components = (AFProduct **)malloc(
           sizeof(AFProduct *) * TotalAFComponents)) == NULL) {
    printf("#fAF: Not enough memory for AFProduct Array!");
    exit(EXIT_FAILURE);
  }

  // Generating AFProduct Data
  // Loop over the AF Records corresponding to each operand and for each operand:
  for (int OperandIndex = 0; OperandIndex < NumOperands; ++OperandIndex) {
#if FAF_DEBUG >= 3
    printf("\t\tOperandIndex: %d\n", OperandIndex);
#endif

    // Create a new AFProduct and copy the ACItem and set the AF.
    AFProduct *NewAFComponent = NULL;

    if (AFItemWRTOperands[OperandIndex] != NULL) {
      assert((*AFItemWRTOperands[OperandIndex])->Components[0] != NULL);
      // Change status of the children of this AFItem to NOT root nodes.
      (*AFItemWRTOperands[OperandIndex])->RootNode = 0;
      double GreatestAF =
          (*AFItemWRTOperands[OperandIndex])->Components[0]->AFs[0] *
          (*AC)->ACWRTOperands[OperandIndex];
      int CandidateIndex = 0;

      // Loop through all Components of this child. Do not skip 0th component to
      //  optimize getting the greatest AF or you will miss some components.
      for (int AFItemsChildComponentIndex = 0;
           AFItemsChildComponentIndex <
           (*AFItemWRTOperands[OperandIndex])->NumAFComponents;
           ++AFItemsChildComponentIndex) {
        // Get the greatest AF and index of the corresponding component.
        if (GreatestAF < (*AFItemWRTOperands[OperandIndex])
                                 ->Components[AFItemsChildComponentIndex]
                                 ->AFs[0] *
                             (*AC)->ACWRTOperands[OperandIndex]) {
          GreatestAF = (*AFItemWRTOperands[OperandIndex])
                           ->Components[AFItemsChildComponentIndex]
                           ->AFs[0] *
                       (*AC)->ACWRTOperands[OperandIndex];
          CandidateIndex = AFItemsChildComponentIndex;
        }
      }

      // Create a new AFProduct and copy the corresponding AFProduct from
      // the AFItem for that operand and also the ACItem and set the AF.
      fAFCreateAFProduct(&NewAFComponent, AFComponentCounter, 1);
      NewAFComponent->Inputs[0] = (*AFItemWRTOperands[OperandIndex])
                                      ->Components[CandidateIndex]
                                      ->Inputs[0];
      NewAFComponent->Factors[0] = *AC;
      NewAFComponent->ACWRTs[0] = OperandIndex;
      NewAFComponent->ProductTails[0] =
          (*AFItemWRTOperands[OperandIndex])->Components[CandidateIndex];
      NewAFComponent->AFs[0] = GreatestAF;
      NewAFComponent->Height = (*AFItemWRTOperands[OperandIndex])
                                   ->Components[CandidateIndex]
                                   ->Height +
                               1;
      NewAFComponent->NumberOfInputs++;
      AFComponentCounter++;

      // Update the statistics.
      updateStatistics(Stats,
                       GreatestAF,
                       (*AC)->ACWRTOperands[OperandIndex],
                       GreatestAF/((*AFItemWRTOperands[OperandIndex])
                                       ->Components[CandidateIndex]
                                       ->Height + 1));
    } else {
      fAFCreateAFProduct(&NewAFComponent, AFComponentCounter,
                         NumFunctionEvaluations);
      NewAFComponent->Inputs[NewAFComponent->NumberOfInputs] =
          (*AC)->OperandNames[OperandIndex];
      NewAFComponent->Factors[NewAFComponent->NumberOfInputs] = *AC;
      NewAFComponent->ACWRTs[NewAFComponent->NumberOfInputs] = OperandIndex;
      NewAFComponent->AFs[NewAFComponent->NumberOfInputs] =
          (*AC)->ACWRTOperands[OperandIndex];
      NewAFComponent->NumberOfInputs++;
      AFComponentCounter++;

      // Update the statistics.
      updateStatistics(Stats,
                       (*AC)->ACWRTOperands[OperandIndex],
                       (*AC)->ACWRTOperands[OperandIndex],
                       (*AC)->ACWRTOperands[OperandIndex]);
    }

    // Add this new AFProduct to the new AFItem
    UpdatingAFItem->Components[UpdatingAFItem->NumAFComponents] =
        NewAFComponent;
    UpdatingAFItem->NumAFComponents++;

#if FAF_DEBUG >= 3
    fAFPrintAFProduct(NewAFComponent);
    printf("\n");
#endif
  }

  //  Add AFItem to the AFTable
  AFs->AFItems[AFs->ListLength] = UpdatingAFItem;
  AFs->ListLength++;
  InstructionExecutionCounter++;

#if FAF_DEBUG >= 2
  printf("\n\tAFItem Created\n");
  printf("\t\tAFId: %d\n", AFs->AFItems[AFs->ListLength - 1]->ItemId);
  printf("\t\tNumComponents: %d\n",
         AFs->AFItems[AFs->ListLength - 1]->NumAFComponents);
  printf("\n");
#endif

  //  Return the AFItem
  return &AFs->AFItems[AFs->ListLength - 1];
}
#else
AFItem **fAFComputeAF(ACItem **AC, AFItem ***AFItemWRTOperands,
                      int NumOperands,
                      int FunctionEvaluationId,
                      int InstructionId,
                      int NumFunctionEvaluations) {
#if FAF_DEBUG >= 2
  printf("\n\tCreating AFItem\n");
  printf("\t\tACId: %d\n", (*AC)->ItemId);
  printf("\t\tFunctionEvaluationId: %d\n", FunctionEvaluationId);
  printf("\t\tInstructionId: %d\n", InstructionId);
  printf("\t\tInstructionExecutionId: %d\n", InstructionExecutionCounter);
  printf("\t\tNumber of Operands: %d\n", NumOperands);
  printf("\t\tRootNode: %s\n", (*AC)->ResultVar);
  printf("\t\tComponents:\n");
  for (int OperandIndex = 0; OperandIndex < NumOperands; ++OperandIndex) {
    if (AFItemWRTOperands[OperandIndex] != NULL)
      printf("\t\t\tAFId %d #Components: %d\n",
             (*AFItemWRTOperands[OperandIndex])->ItemId,
             (*AFItemWRTOperands[OperandIndex])->NumAFComponents);
    else
      printf("\t\t\tAFId %d #Components: 0\n", -1);
  }

  printf("\n");
#endif

  // This determines whether we create a new AFItem or update an existing one.
  int NewAFItem = 0;

  // Find an existing AFItem with the same InstructionId and InstructionExecutionId or create a new one.
  int AFItemIndex = fAFGetAFItemFromList(AFs->AFItems, AFs->ListLength,
                                         InstructionId, InstructionExecutionCounter);
  AFItem *UpdatingAFItem = NULL;
  if (AFItemIndex == -1) {
#if FAF_DEBUG >= 3
    printf("\t\t\tCreating a new AFItem\n");
#endif
    NewAFItem = 1;

    fAFCreateAFItem(&AFs->AFItems[AFs->ListLength], AFItemCounter, FunctionEvaluationId,
                    InstructionId, InstructionExecutionCounter, 0);
    AFs->ListLength++;
    AFItemIndex = AFs->ListLength - 1;
    UpdatingAFItem = AFs->AFItems[AFs->ListLength - 1];
    AFItemCounter++;
  } else
    UpdatingAFItem = AFs->AFItems[AFItemIndex];

  int TotalAFComponents = 0;

  // Computing the number of Components/AFPaths that contribute to the Relative
  // Error of this instruction.
  for (int OperandIndex = 0; OperandIndex < NumOperands; ++OperandIndex) {
    if (AFItemWRTOperands[OperandIndex] != NULL)
      TotalAFComponents += (*AFItemWRTOperands[OperandIndex])->NumAFComponents;
    else
      TotalAFComponents++;
  }

#if FAF_DEBUG >= 3
  printf("\t\tTotalAFComponents: %d", TotalAFComponents);
  printf("\t\tNewAFItem: %d\n\n", NewAFItem);
#endif

  if (NewAFItem) {
    if ((UpdatingAFItem->Components = (AFProduct **)malloc(
             sizeof(AFProduct *) * TotalAFComponents)) == NULL) {
      printf("#fAF: Not enough memory for AFProduct Array!");
      exit(EXIT_FAILURE);
    }
  }

  int AFItemComponentIndex = 0;

  // Generating AFProduct Data
  // Loop over the AF Records corresponding to each operand and for each operand:
  for (int OperandIndex = 0; OperandIndex < NumOperands; ++OperandIndex) {
#if FAF_DEBUG >= 3
    printf("\t\tOperandIndex: %d\n", OperandIndex);
#endif
    if (AFItemWRTOperands[OperandIndex] != NULL) {
      if(NewAFItem)
        // Change status of the children of this AFItem to NOT root nodes.
        (*AFItemWRTOperands[OperandIndex])->RootNode = 0;

      // Loop through all Components of this child.
      for (int AFItemsChildComponentIndex = 0;
           AFItemsChildComponentIndex <
           (*AFItemWRTOperands[OperandIndex])->NumAFComponents;
           ++AFItemsChildComponentIndex) {
        // Create a new AFProduct and copy the corresponding AFProduct from
        // the AFItem for that operand and also the ACItem and set the AF.
        AFProduct *UpdatingAFComponent = NULL;

        if (NewAFItem) {
          fAFCreateAFProduct(&UpdatingAFComponent, AFComponentCounter,
                             NumFunctionEvaluations);
          AFComponentCounter++;
        } else
          UpdatingAFComponent =
              UpdatingAFItem->Components[AFItemComponentIndex];

        UpdatingAFComponent->Inputs =
            (*AFItemWRTOperands[OperandIndex])
                ->Components[AFItemsChildComponentIndex]
                ->Inputs;
        UpdatingAFComponent->Factors[UpdatingAFComponent->NumberOfInputs] = *AC;
        UpdatingAFComponent->ACWRTs[UpdatingAFComponent->NumberOfInputs] = OperandIndex;
        UpdatingAFComponent->ProductTails[UpdatingAFComponent->NumberOfInputs] =
            (*AFItemWRTOperands[OperandIndex])
                ->Components[AFItemsChildComponentIndex];
        UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs] =
            (*AFItemWRTOperands[OperandIndex])
                ->Components[AFItemsChildComponentIndex]
                ->AFs[(*AFItemWRTOperands[OperandIndex])
                          ->Components[AFItemsChildComponentIndex]
                          ->NumberOfInputs-1] *
            (*AC)->ACWRTOperands[OperandIndex];
        UpdatingAFComponent->Height =
            (*AFItemWRTOperands[OperandIndex])
                ->Components[AFItemsChildComponentIndex]
                ->Height +
            1;
        UpdatingAFComponent->NumberOfInputs++;

        if (NewAFItem) {
          // Add this new AFProduct to the new AFItem
          UpdatingAFItem->Components[UpdatingAFItem->NumAFComponents] =
              UpdatingAFComponent;
          UpdatingAFItem->NumAFComponents++;
        }

#if FAF_DEBUG >= 3
        printf("\t\tAFItem's Child's Component Number: %d\n", AFItemsChildComponentIndex);
        printf("\t\tAFItem's Component Number: %d\n", AFItemComponentIndex);
        fAFPrintAFProduct(UpdatingAFComponent);
        printf("\n");
#endif

        AFItemComponentIndex++;

        // Update the statistics
        updateStatistics(Stats,
                         UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs-1],
                         (*AC)->ACWRTOperands[OperandIndex],
                         UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs-1]/UpdatingAFComponent->Height);
      }
    } else {
      // Create a new AFProduct and copy the ACItem and set the AF.
      AFProduct *UpdatingAFComponent = NULL;

      if (NewAFItem) {
        fAFCreateAFProduct(&UpdatingAFComponent, AFComponentCounter,
                           NumFunctionEvaluations);
        AFComponentCounter++;
      } else
        UpdatingAFComponent = UpdatingAFItem->Components[AFItemComponentIndex];
      UpdatingAFComponent->Inputs[UpdatingAFComponent->NumberOfInputs] =
          (*AC)->OperandNames[OperandIndex];
      UpdatingAFComponent->Factors[UpdatingAFComponent->NumberOfInputs] = *AC;
      UpdatingAFComponent->ACWRTs[UpdatingAFComponent->NumberOfInputs] = OperandIndex;
      UpdatingAFComponent->ProductTails[UpdatingAFComponent->NumberOfInputs] =
          NULL;
      UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs] =
          (*AC)->ACWRTOperands[OperandIndex];
      UpdatingAFComponent->NumberOfInputs++;

      if (NewAFItem) {
        // Add this new AFProduct to the new AFItem
        UpdatingAFItem->Components[UpdatingAFItem->NumAFComponents] =
            UpdatingAFComponent;
        UpdatingAFItem->NumAFComponents++;
      }

#if FAF_DEBUG >= 3
      printf("\t\tAFItem's Component Number: %d\n", AFItemComponentIndex);
      fAFPrintAFProduct(UpdatingAFComponent);
      printf("\n");
#endif

      AFItemComponentIndex++;

      // Update the statistics
      updateStatistics(Stats,
                       UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs-1],
                       (*AC)->ACWRTOperands[OperandIndex],
                       UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs-1]);
    }
  }

  InstructionExecutionCounter++;

#if FAF_DEBUG >= 2
  if(NewAFItem)
    printf("\n\tAFItem Created");
  printf("\n\t\tAFId: %d\n", AFs->AFItems[AFItemIndex]->ItemId);
  printf("\t\tNumComponents: %d\n",
         AFs->AFItems[AFItemIndex]->NumAFComponents);
  printf("\n");
#endif

  //  Return the AFItem
  return &AFs->AFItems[AFItemIndex];
}
#endif

// This function prints the ItemId and LineNumber corresponding to the ACItems
// in the flattenedProductTail at index M of an AFProduct
void fAFPrintAFProductSources(AFProduct *AFProduct, int M) {
  printf("(%d, %d)", AFProduct->Factors[M]->ItemId, AFProduct->Factors[M]->LineNumber);
  if(AFProduct->ProductTails[0] != NULL) {
    printf(", ");
    fAFPrintAFProductSources(AFProduct->ProductTails[M], M);
  }
}


void fAFPrintTopAmplificationPaths() {
  printf("Printing Top Amplification Paths from Last AFItem\n");
  // Sorting the list according to Amplification Factors
  qsort(AFs->AFItems[AFs->ListLength-1]->Components, AFs->AFItems[AFs->ListLength-1]->NumAFComponents,
        sizeof(AFProduct *), fAFComparator);

  // Printing Results
  printf("\n");
  printf("The top Amplification Paths are:\n");
  for (int I = 0; I < min(5, AFs->AFItems[AFs->ListLength-1]->NumAFComponents); ++I) {
    int M = findMaxIndex(Paths[I]->AFs, Paths[I]->NumberOfInputs);

    printf("AF: %0.15lf (ULPErr: %lf; %lf digits) of Node with AFId:%d WRT Input:%s through path: [",
           AFs->AFItems[AFs->ListLength-1]->Components[I]->AFs[M],
           ceil(log2(AFs->AFItems[AFs->ListLength-1]->Components[I]->AFs[M])),
           ceil(log10(log2(AFs->AFItems[AFs->ListLength-1]->Components[I]->AFs[M]))),
           AFs->AFItems[AFs->ListLength-1]->Components[I]->ItemId,
           AFs->AFItems[AFs->ListLength-1]->Components[I]->Inputs[M]);

    fAFPrintAFProductSources(AFs->AFItems[AFs->ListLength-1]->Components[I], M);
    printf("]\n");
  }
  printf("\n");
  printf("Printed Top Amplification Paths\n");
}

void fAFPrintTopFromAllAmplificationPaths() {
  printf("Printing Top Amplification Paths Over ALL Paths\n");
  fAFFlattenAllComponentPaths();
  // Sorting the list according to Amplification Factors
  qsort(Paths, AFComponentCounter,sizeof(AFProduct *), fAFComparator);

//  for (uint64_t I = 0; I < AFComponentCounter; ++I) {
//    fAFPrintAFProduct(Paths[I]);
//    printf("\n");
//  }

  // Printing Results
  printf("\n");
  printf("The top Amplification Paths are:\n");
  for (int I = 0; I < min(10, AFComponentCounter); ++I) {
    int M = findMaxIndex(Paths[I]->AFs, Paths[I]->NumberOfInputs);

    printf("AF: %0.15lf (ULPErr: %lf; %lf digits) of Node with AFId:%d WRT Input:%s through path: [",
           Paths[I]->AFs[M],
           ceil(log2(Paths[I]->AFs[M])),
           ceil(log10(log2(Paths[I]->AFs[M]))),
           Paths[I]->ItemId,
           Paths[I]->Inputs[M]);

    fAFPrintAFProductSources(Paths[I], M);
    printf("]\n");
  }
  printf("\n");
  printf("Printed Top Amplification Paths\n");

  char File[5000];
  fAFGenerateFileString(File, "SortedAFs_", ".json");

  // Table Output
  FILE *FP;
  if((FP = fopen(File, "w")) != NULL) {
    fAFStoreAFProducts(FP, Paths, AFComponentCounter);
    fclose(FP);
  } else {
    printf("%s cannot be opened.\n", File);
  }

  printf("Ranked AF Paths written to file: %s\n", File);
}

// Prints the Statistics of the Analysis
void fAFPrintStatistics(int NumFunctionEvaluations) {
#if FAF_DEBUG
  printf("\nPrinting Statistics\n");
#endif

  printf("Number of Computation DAGs\t\t\t: %d\n", getNumberOfComputationDAGs(AFs));
  printf("Number of Computation Paths\t\t\t: %d\n", getNumberOfComputationPaths(AFs));
  printf("Number of Floating-Point Operations\t: %lu\n", AFs->ListLength);
  printf("Average Amplification Factor\t\t: %0.15lf\n", getAverageAmplificationFactor(AFs, NumFunctionEvaluations));

  printStatistics(Stats);

  statisticsDelete(Stats);

#if FAF_DEBUG
  printf("\nPrinted Statistics\n");
#endif
}


void fAFStoreAFs() {
  printf("\nWriting Amplification Factors to file.\n");
  // Generate a file path + file name string to store the AF Records
  char File[5000];
  fAFGenerateFileString(File, "fAF_", ".json");

  // Table Output
  FILE *FP;
  if((FP = fopen(File, "w")) != NULL) {
    fAFStoreAFItems(FP, AFs->AFItems, AFs->ListLength);
    fclose(FP);
  } else {
    printf("%s cannot be opened.\n", File);
  }

  printf("Amplification Factors written to file: %s\n", File);
}

#endif // LLVM_AMPLIFICATIONFACTOR_H
