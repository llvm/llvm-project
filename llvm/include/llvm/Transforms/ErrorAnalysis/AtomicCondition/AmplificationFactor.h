//
// Created by tanmay on 7/11/22.
//

#ifndef LLVM_AMPLIFICATIONFACTOR_H
#define LLVM_AMPLIFICATIONFACTOR_H

#include "AtomicCondition.h"
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
    1e-6, // Extendin is not a GNU library function. Giving it 1 ULP error
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
    1e-16, // Extendin is not a GNU library function. Giving it 1 ULP error
    1e-16,
};

/*----------------------------------------------------------------------------*/
/* Data Structures, Associated Functions and Types                            */
/*----------------------------------------------------------------------------*/

/* ------------------------------- AFProduct -------------------------------*/

struct AFProduct {
  int ItemId;

  ACItem **Factors;
  struct AFProduct **ProductTails;
  char **Inputs;
  double *AFs;

  // Metadata
  int Height;
  int NumberOfInputs;
};

typedef struct AFProduct AFProduct;

void fAFCreateAFComponent(AFProduct **AddressToAllocateAt,
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
  (*AddressToAllocateAt)->Factors[0] = NULL;

  if(((*AddressToAllocateAt)->ProductTails = (AFProduct **)malloc(sizeof(AFProduct *) *
                                                           LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for ProductTails!");
    exit(EXIT_FAILURE);
  }
  (*AddressToAllocateAt)->ProductTails[0] = NULL;

  if(((*AddressToAllocateAt)->Inputs = (char **)malloc(sizeof(char *) *
                                                                   LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for Inputs!");
    exit(EXIT_FAILURE);
  }
  (*AddressToAllocateAt)->Inputs[0] = NULL;

  if(((*AddressToAllocateAt)->AFs = (double *)malloc(sizeof(double *) *
                                                        LengthOfLists)) == NULL) {
    printf("#fAF: Not enough memory for AFs!");
    exit(EXIT_FAILURE);
  }
  (*AddressToAllocateAt)->AFs[0] = 0.0;

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

  printf("\t\t\t\"ACItemStrings\": [\"%s\"",
         ProductObject->Factors[0]->ResultVar);
  for (int I = 1; I < ProductObject->NumberOfInputs; ++I) {
    printf(",\"%s\"", ProductObject->Factors[I]->ResultVar);
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

  printf("\t\t\t\"AFs\": [%0.15lf", ProductObject->AFs[0]);
  for (int I = 1; I < ProductObject->NumberOfInputs; ++I) {
    printf(",%0.15lf", ProductObject->AFs[I]);
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

    fprintf(FP, "\t\t\t\"ACItemIds\": [%d", ObjectPointerList[J]->Factors[0]->ItemId);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",%d", ObjectPointerList[J]->Factors[I]->ItemId);
    }
    fprintf(FP, "],\n");

    fprintf(FP, "\t\t\t\"ACItemStrings\": [\"%s\"", ObjectPointerList[J]->Factors[0]->ResultVar);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",\"%s\"", ObjectPointerList[J]->Factors[I]->ResultVar);
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

    fprintf(FP, "\t\t\t\"AFs\": [%0.15lf", ObjectPointerList[J]->AFs[0]);
    for (int I = 1; I < ObjectPointerList[J]->NumberOfInputs; ++I) {
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
  int ItemId;
  int InstructionId;
  int ExecutionId;

  AFProduct** Components;
  int NumAFComponents;
};

typedef struct AFItem AFItem;

void fAFCreateAFItem(AFItem **AddressToAllocateAt,
                     int ItemId,
                     int InstructionId,
                     int ExecutionId,
                     int NumAFComponents) {
  if((*AddressToAllocateAt = (AFItem *)malloc(sizeof(AFItem))) == NULL) {
    printf("#fAF: Not enough memory for AFItem!");
    exit(EXIT_FAILURE);
  }

  (*AddressToAllocateAt)->ItemId = ItemId;
  (*AddressToAllocateAt)->InstructionId = InstructionId;
  (*AddressToAllocateAt)->ExecutionId = ExecutionId;
  (*AddressToAllocateAt)->Components = NULL;
  (*AddressToAllocateAt)->NumAFComponents = NumAFComponents;

  return ;
}

// Gets the AFItem with InstructionId and ExecutionId from List of AFItems.
AFItem *fAFGetAFItemFromList(AFItem **AFItemList,
                             int NumAFItems,
                             int InstructionId,
                             int ExecutionId) {
  for (int I = 0; I < NumAFItems; ++I)
    if (AFItemList[I]->InstructionId == InstructionId &&
        AFItemList[I]->ExecutionId == ExecutionId)
      return AFItemList[I];

  return NULL;
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

      fprintf(FP, "\t\t\t\"ACItemStrings\": [\"%s\"", ObjectPointerList[K]->Components[J]->Factors[0]->ResultVar);
      for (int I = 1; I < ObjectPointerList[K]->Components[J]->NumberOfInputs; ++I) {
        fprintf(FP, ",\"%s\"", ObjectPointerList[K]->Components[J]->Factors[I]->ResultVar);
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

      fprintf(FP, "\t\t\t\"AFs\": [%0.15lf", ObjectPointerList[K]->Components[J]->AFs[0]);
      for (int I = 1; I < ObjectPointerList[K]->Components[J]->NumberOfInputs; ++I) {
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
  if((*AddressToAllocateAt = (AFTable *)malloc(sizeof(AFTable))) == NULL) {
    printf("#fAF: Not enough memory for AFItem!");
    exit(EXIT_FAILURE);
  }

  return ;
}

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/

int AFItemCounter;
int AFComponentCounter;
int PlotDataCounter;
int ExecutionCounter;
AFTable *AFs;
AFProduct **Paths;

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

    fprintf(FP, "\t\t\t\"ACItemStrings\": [\"%s\"", (*ObjectToStore)->Components[J]->Factors[0]->ResultVar);
    for (int I = 1; I < (*ObjectToStore)->Components[J]->NumberOfInputs; ++I) {
      fprintf(FP, ",\"%s\"", (*ObjectToStore)->Components[J]->Factors[I]->ResultVar);
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
    if((*(AFProduct **)A)->ProductTails[I]!=NULL &&
        fabs((*(AFProduct **)A)->AFs[I]) > AF1)
      AF1 = fabs((*(AFProduct **)A)->AFs[I]);
  }
  double AF2 = fabs((*(AFProduct **)B)->AFs[0]);
  for (int I = 0; I < (*(AFProduct **)B)->NumberOfInputs; ++I) {
    if((*(AFProduct **)B)->ProductTails[I]!=NULL &&
            fabs((*(AFProduct **)B)->AFs[I]) > AF2)
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

  return ;
}

/*----------------------------------------------------------------------------*/
/* Analysis Functions                                                         */
/*----------------------------------------------------------------------------*/

// Inputs: AC Record corresponding to the current instruction and the AF Records
//  corresponding the operands.
// Note: The current instruction information can be found in the AC record.
// Steps:
//  Create a new AF record and initialize data members and allocate memory for
//    AFComponents array.
//  Loop over the AF Records corresponding to each operand and for each operand:
//    If AFRecord is not NULL:
//      Loop over the AFProduct for this AF Record and for each Component
//        Create a new AFProduct and copy the corresponding AFProduct from the
//        AFItem for that operand and also the ACItem and set the AF.
//        Add this new AFProduct to the new AFItem
//    Else:
//      Create a new AFProduct and copy the ACItem and set the AF.
//      Add this new AFProduct to the new AFItem
//  Add AFItem to the AFTable
//  Return the AFItem
#if MEMORY_OPT
AFItem **fAFComputeAF(ACItem **AC, AFItem ***AFItemWRTOperands,
                      int NumOperands,
                      int InstructionId,
                      int NumFunctionEvaluations) {
#if FAF_DEBUG>=2
  printf("\nCreating AFItem\n");
  printf("\tACId: %d\n", (*AC)->ItemId);
  printf("\tInstructionId: %d\n", InstructionId);
  printf("\tExecutionId: %d\n", ExecutionCounter);
  printf("\tNumber of Operands: %d\n", NumOperands);
  printf("\tRootNode: %s\n", (*AC)->ResultVar);
  printf("\tComponents:\n");
  for (int I = 0; I < NumOperands; ++I) {
    if(AFItemWRTOperands[I] != NULL)
      printf("\t\tAFId %d #Components: %d\n",
             (*AFItemWRTOperands[I])->ItemId,
             (*AFItemWRTOperands[I])->NumAFComponents);
    else
      printf("\t\tAFId %d #Components: 0\n", -1);
  }

  printf("\n");
#endif

  //  Create a new AF record and initialize data members and allocate memory for
  //  AFComponents array.
  AFItem *UpdatingAFItem = NULL;

  fAFCreateAFItem(&UpdatingAFItem,
                  AFItemCounter,
                  InstructionId,
                  ExecutionCounter,
                  0);
  AFItemCounter++;

  int TotalAFComponents = 0;
  TotalAFComponents+=NumOperands;

#if FAF_DEBUG>=2
  printf("\tTotalAFComponents: %d\n\n", TotalAFComponents);
#endif

  if ((UpdatingAFItem->Components = (AFProduct **)malloc(
           sizeof(AFProduct *) * TotalAFComponents)) == NULL) {
    printf("#fAF: Not enough memory for AFProduct Array!");
    exit(EXIT_FAILURE);
  }

  // Generating AFProduct Data
  // Loop over the AF Records corresponding to each operand and for each operand:
  for (int I = 0; I < NumOperands; ++I) {
    // Create a new AFProduct and copy the ACItem and set the AF.
    AFProduct *NewAFComponent = NULL;

    if(AFItemWRTOperands[I] != NULL) {
      assert((*AFItemWRTOperands[I])->Components[0] != NULL);
      double GreatestAF = (*AFItemWRTOperands[I])->Components[0]->AFs[0] *
                          (*AC)->ACWRTOperands[I];
      int CandidateIndex = 0;

      // Loop through all Components of this child. Do not skip 0th component to
      //  optimize getting the greatest AF or you will miss some components.
      for (int J = 0; J < (*AFItemWRTOperands[I])->NumAFComponents; ++J) {
        // Get the greatest AF and index of the corresponding component.
        if (GreatestAF < (*AFItemWRTOperands[I])->Components[J]->AFs[0] *
                             (*AC)->ACWRTOperands[I]) {
          GreatestAF = (*AFItemWRTOperands[I])->Components[J]->AFs[0] *
                       (*AC)->ACWRTOperands[I];
          CandidateIndex = J;
        }
      }

      // Create a new AFProduct and copy the corresponding AFProduct from
      // the AFItem for that operand and also the ACItem and set the AF.
      fAFCreateAFComponent(&NewAFComponent,
                           AFComponentCounter,
                           1);
      NewAFComponent->Inputs[0] = (*AFItemWRTOperands[I])->Components[CandidateIndex]->Inputs[0];
      NewAFComponent->Factors[0] = *AC;
      NewAFComponent->ProductTails[0] = (*AFItemWRTOperands[I])->Components[CandidateIndex];
      NewAFComponent->AFs[0] = (*AFItemWRTOperands[I])->Components[CandidateIndex]->AFs[0] *
                               (*AC)->ACWRTOperands[I];
      NewAFComponent->Height = (*AFItemWRTOperands[I])->Components[CandidateIndex]->Height+1;
      NewAFComponent->NumberOfInputs++;
      AFComponentCounter++;
    } else {
      fAFCreateAFComponent(&NewAFComponent,
                           AFComponentCounter,
                           NumFunctionEvaluations);
      NewAFComponent->Inputs[NewAFComponent->NumberOfInputs] = (*AC)->OperandNames[I];
      NewAFComponent->Factors[NewAFComponent->NumberOfInputs] = *AC;
      NewAFComponent->AFs[NewAFComponent->NumberOfInputs] = (*AC)->ACWRTOperands[I];
      NewAFComponent->NumberOfInputs++;
      AFComponentCounter++;
    }

    // Add this new AFProduct to the new AFItem
    UpdatingAFItem->Components[UpdatingAFItem->NumAFComponents] = NewAFComponent;
    UpdatingAFItem->NumAFComponents++;
  }

  //  Add AFItem to the AFTable
  AFs->AFItems[AFs->ListLength] = UpdatingAFItem;
  AFs->ListLength++;
  ExecutionCounter++;

#if FAF_DEBUG>=2
  printf("\nAFItem Created\n");
  printf("\tAFId: %d\n", AFs->AFItems[AFs->ListLength-1]->ItemId);
  printf("\tNumComponents: %d\n", AFs->AFItems[AFs->ListLength-1]->NumAFComponents);
  printf("\n");
#endif

  //  Return the AFItem
  return &AFs->AFItems[AFs->ListLength-1];
}
#else
AFItem **fAFComputeAF(ACItem **AC, AFItem ***AFItemWRTOperands,
                      int NumOperands,
                      int InstructionId,
                      int NumFunctionEvaluations) {
#if FAF_DEBUG>=2
  printf("\nCreating AFItem\n");
  printf("\tACId: %d\n", (*AC)->ItemId);
  printf("\tInstructionId: %d\n", InstructionId);
  printf("\tExecutionId: %d\n", ExecutionCounter);
  printf("\tNumber of Operands: %d\n", NumOperands);
  printf("\tRootNode: %s\n", (*AC)->ResultVar);
  printf("\tComponents:\n");
  for (int I = 0; I < NumOperands; ++I) {
    if(AFItemWRTOperands[I] != NULL)
      printf("\t\tAFId %d #Components: %d\n",
             (*AFItemWRTOperands[I])->ItemId,
             (*AFItemWRTOperands[I])->NumAFComponents);
    else
      printf("\t\tAFId %d #Components: 0\n", -1);
  }

  printf("\n");
#endif

  // This determines whether we create a new AFItem or update an existing one.
  int NewAFItem = 0;

  // Find an existing AFItem with the same InstructionId and ExecutionId or create
  // a new one.
  AFItem *UpdatingAFItem = NULL;
  UpdatingAFItem = fAFGetAFItemFromList(AFs->AFItems,
                                        AFs->ListLength,
                                        InstructionId,
                                        ExecutionCounter);
  if(UpdatingAFItem == NULL) {
    printf("Creating a new AFItem\n");
    NewAFItem = 1;

    fAFCreateAFItem(&UpdatingAFItem, AFItemCounter, InstructionId, ExecutionCounter, 0);
    AFItemCounter++;
  }

  int TotalAFComponents = 0;

  // Computing the number of Components/AFPaths that contribute to the Relative
  // Error of this instruction.
  for (int I = 0; I < NumOperands; ++I) {
    if (AFItemWRTOperands[I] != NULL)
      TotalAFComponents+=(*AFItemWRTOperands[I])->NumAFComponents;
    else
      TotalAFComponents++;
  }

#if FAF_DEBUG>=2
  printf("\tTotalAFComponents: %d\n\n", TotalAFComponents);
  printf("\tNewAFItem: %d\n", NewAFItem);
#endif

  if (NewAFItem) {
    if ((UpdatingAFItem->Components = (AFProduct **)malloc(
             sizeof(AFProduct *) * TotalAFComponents)) == NULL) {
      printf("#fAF: Not enough memory for AFProduct Array!");
      exit(EXIT_FAILURE);
    }
  }

  int K = 0;

  // Generating AFProduct Data
  // Loop over the AF Records corresponding to each operand and for each operand:
  for (int I = 0; I < NumOperands; ++I) {
    if(AFItemWRTOperands[I] != NULL) {

      // Loop through all Components of this child.
      for (int J = 0; J < (*AFItemWRTOperands[I])->NumAFComponents; ++J) {
        // Create a new AFProduct and copy the corresponding AFProduct from
        // the AFItem for that operand and also the ACItem and set the AF.
        AFProduct *UpdatingAFComponent = NULL;

        if (NewAFItem) {
          fAFCreateAFComponent(&UpdatingAFComponent, AFComponentCounter,
                               NumFunctionEvaluations);
          AFComponentCounter++;
        }
        else
          UpdatingAFComponent = UpdatingAFItem->Components[K];

        UpdatingAFComponent->Inputs = (*AFItemWRTOperands[I])->Components[J]->Inputs;
        UpdatingAFComponent->Factors[UpdatingAFComponent->NumberOfInputs] = *AC;
        UpdatingAFComponent->ProductTails[UpdatingAFComponent->NumberOfInputs] = (*AFItemWRTOperands[I])->Components[J];
        UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs] =
            (*AFItemWRTOperands[I])->Components[J]->AFs[(*AFItemWRTOperands[I])->Components[J]->NumberOfInputs] *
            (*AC)->ACWRTOperands[I];
        UpdatingAFComponent->Height = (*AFItemWRTOperands[I])->Components[J]->Height+1;
        UpdatingAFComponent->NumberOfInputs++;

        if(NewAFItem) {
          // Add this new AFProduct to the new AFItem
          UpdatingAFItem->Components[UpdatingAFItem->NumAFComponents] =
              UpdatingAFComponent;
          UpdatingAFItem->NumAFComponents++;
        }

//        printf("J: %d\n", J);
//        fAFPrintAFProduct(UpdatingAFComponent);
//        printf("\n");

        K++;
      }
    } else {
      // Create a new AFProduct and copy the ACItem and set the AF.
      AFProduct *UpdatingAFComponent = NULL;

      if (NewAFItem) {
        fAFCreateAFComponent(&UpdatingAFComponent, AFComponentCounter,
                             NumFunctionEvaluations);
        AFComponentCounter++;
      }
      else
        UpdatingAFComponent = UpdatingAFItem->Components[K];
      UpdatingAFComponent->Inputs[UpdatingAFComponent->NumberOfInputs] = (*AC)->OperandNames[I];
      UpdatingAFComponent->Factors[UpdatingAFComponent->NumberOfInputs] = *AC;
      UpdatingAFComponent->ProductTails[UpdatingAFComponent->NumberOfInputs] = NULL;
      UpdatingAFComponent->AFs[UpdatingAFComponent->NumberOfInputs] = (*AC)->ACWRTOperands[I];
      UpdatingAFComponent->NumberOfInputs++;

      if(NewAFItem) {
        // Add this new AFProduct to the new AFItem
        UpdatingAFItem->Components[UpdatingAFItem->NumAFComponents] =
            UpdatingAFComponent;
        UpdatingAFItem->NumAFComponents++;
      }
//      fAFPrintAFProduct(UpdatingAFComponent);
//      printf("\n");

      K++;
    }
  }

  //  Add AFItem to the AFTable
  AFs->AFItems[AFs->ListLength] = UpdatingAFItem;
  AFs->ListLength++;
  ExecutionCounter++;

#if FAF_DEBUG>=2
  printf("\nAFItem Created\n");
  printf("\tAFId: %d\n", AFs->AFItems[AFs->ListLength-1]->ItemId);
  printf("\tNumComponents: %d\n", AFs->AFItems[AFs->ListLength-1]->NumAFComponents);
  printf("\n");
#endif

  //  Return the AFItem
  return &AFs->AFItems[AFs->ListLength-1];
}
#endif

// This function prints the ItemId and LineNumber corresponding to the ACItems
// in the flattenedProductTail of an AFProduct
void fAFPrintAFProductSources(AFProduct *AFProduct) {
  printf("(%d, %d)", AFProduct->Factors[0]->ItemId, AFProduct->Factors[0]->LineNumber);
  if(AFProduct->ProductTails[0] != NULL) {
    printf(", ");
    fAFPrintAFProductSources(AFProduct->ProductTails[0]);
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
    printf("AF: %0.15lf (ULPErr: %lf; %lf digits) of Node with AFId:%d WRT Input:%s through path: [",
           AFs->AFItems[AFs->ListLength-1]->Components[I]->AFs[0],
           ceil(log2(AFs->AFItems[AFs->ListLength-1]->Components[I]->AFs[0])),
           ceil(log10(log2(AFs->AFItems[AFs->ListLength-1]->Components[I]->AFs[0]))),
           AFs->AFItems[AFs->ListLength-1]->Components[I]->ItemId,
           AFs->AFItems[AFs->ListLength-1]->Components[I]->Inputs[0]);

    fAFPrintAFProductSources(AFs->AFItems[AFs->ListLength-1]->Components[I]);
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
    printf("AF: %0.15lf (ULPErr: %lf; %lf digits) of Node with AFId:%d WRT Input:%s through path: [",
           Paths[I]->AFs[0],
           ceil(log2(Paths[I]->AFs[0])),
           ceil(log10(log2(Paths[I]->AFs[0]))),
           Paths[I]->ItemId,
           Paths[I]->Inputs[0]);

    fAFPrintAFProductSources(Paths[I]);
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


void fAFStoreAFs() {
#if NO_DATA_DUMP
#else
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
#endif
}

#endif // LLVM_AMPLIFICATIONFACTOR_H
