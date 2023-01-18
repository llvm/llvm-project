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
  ACItem *Factor;
  struct AFProduct *ProductTail;
  char *Input;
  int Height;
  double AF;
};

typedef struct AFProduct AFProduct;

void fAFCreateAFComponent(AFProduct **AddressToAllocateAt) {
  if((*AddressToAllocateAt = (AFProduct *)malloc(sizeof(AFProduct))) == NULL) {
    printf("#fAF: Not enough memory for AFProduct!");
    exit(EXIT_FAILURE);
  }

  return ;
}

AFProduct **fAFFlattenAFComponentsPath(AFProduct *ProductObject) {
  AFProduct *ProductObjectWalker = ProductObject;
  AFProduct **ProductPath;

  if((ProductPath =
           (AFProduct **)malloc(sizeof(AFProduct*) * ProductObject->Height)) == NULL) {
    printf("#fAF: Not enough memory for AFProduct pointers!");
    exit(EXIT_FAILURE);
  }
  ProductPath[0] = &*ProductObject;

  for (int I = 1; I < ProductObject->Height; ++I) {
    ProductPath[I] = &*(ProductObjectWalker->ProductTail);
    ProductObjectWalker = ProductObjectWalker->ProductTail;
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
    if (fprintf(FP,
                "\t\t{\n"
                "\t\t\t\"ProductItemId\": %d,\n"
                "\t\t\t\"ACItemId\":%d,\n"
                "\t\t\t\"ACItemString\": \"%s\",\n"
                "\t\t\t\"ProductTailItemId\": %d,\n"
                "\t\t\t\"Input\": \"%s\",\n"
                "\t\t\t\"AF\": %lf,\n",
                ObjectPointerList[J]->ItemId,
                ObjectPointerList[J]->Factor->ItemId,
                ObjectPointerList[J]->Factor->ResultVar,
                ObjectPointerList[J]->
                        ProductTail!=NULL?ObjectPointerList[J]->ProductTail->ItemId:
                    -1,
                ObjectPointerList[J]->Input,
                ObjectPointerList[J]->AF) > 0) {

      fprintf(FP, "\t\t\t\"Path(AFProductIds)\": [%d", ProductPath[0]->ItemId);
      for (int K = 1; K < ObjectPointerList[J]->Height; ++K)
        fprintf(FP, ", %d", ProductPath[K]->ItemId);

      if ((uint64_t)J == NumObjects-1)
        fprintf(FP, "]\n\t\t}\n");
      else
        fprintf(FP, "]\n\t\t},\n");
    }
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");
}

/* ---------------------------------- AFItem ---------------------------------*/

struct AFItem {
  int ItemId;
  AFProduct** Components;
  int NumAFComponents;
};

typedef struct AFItem AFItem;

void fAFCreateAFItem(AFItem **AddressToAllocateAt) {
  if((*AddressToAllocateAt = (AFItem *)malloc(sizeof(AFItem))) == NULL) {
    printf("#fAF: Not enough memory for AFItem!");
    exit(EXIT_FAILURE);
  }

  return ;
}

// Function to write NumObjects number of AFItems from ObjectPointerList into
// file with descriptor FP.
void fAFStoreAFItems(FILE *FP, AFItem **ObjectPointerList, uint64_t NumObjects) {
  assert(FP != NULL && "File descriptor is NULL.");
  assert(ObjectPointerList != NULL && "ObjectPointerList is NULL");

  fprintf(FP, "{\n");

  fprintf(FP, "\t\"AFs\": [\n");
  int I = 0;
  while ((uint64_t)I < NumObjects) {
    for (int J = 0; J < ObjectPointerList[I]->NumAFComponents; ++J) {
      AFProduct **ProductPath= fAFFlattenAFComponentsPath(ObjectPointerList[I]->Components[J]);
      if (fprintf(FP,
                  "\t\t{\n"
                  "\t\t\t\"ProductItemId\": %d,\n"
                  "\t\t\t\"ACItemId\":%d,\n"
                  "\t\t\t\"ACItemString\": \"%s\",\n"
                  "\t\t\t\"ProductTailItemId\": %d,\n"
                  "\t\t\t\"Input\": \"%s\",\n"
                  "\t\t\t\"AF\": %0.15lf,\n",
                  ObjectPointerList[I]->Components[J]->ItemId,
                  ObjectPointerList[I]->Components[J]->Factor->ItemId,
                  ObjectPointerList[I]->Components[J]->Factor->ResultVar,
                  ObjectPointerList[I]->Components[J]->
                          ProductTail!=NULL?ObjectPointerList[I]->Components[J]->ProductTail->ItemId:
                      -1,
                  ObjectPointerList[I]->Components[J]->Input,
                  ObjectPointerList[I]->Components[J]->AF) > 0) {

        fprintf(FP, "\t\t\t\"Path(AFProductIds)\": [%d", ProductPath[0]->ItemId);
        for (int K = 1; K < ObjectPointerList[I]->Components[J]->Height; ++K)
          fprintf(FP, ", %d", ProductPath[K]->ItemId);

        if ((uint64_t)I == NumObjects-1 && J == ObjectPointerList[I]->NumAFComponents-1)
          fprintf(FP, "]\n\t\t}\n");
        else
          fprintf(FP, "]\n\t\t},\n");
      }
    }

    I++;

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
AFTable *AFs;
AFProduct **Paths;

/*----------------------------------------------------------------------------*/
/* Utility Functions                                                          */
/*----------------------------------------------------------------------------*/

#pragma clang optimize off
void fAFfp32markForResult(float res) {
  return ;
}

void fAFfp64markForResult(double res) {
  return ;
}

#pragma clang optimize on

int min(int a, int b) {
  return (a > b)? b : a;
}

AFProduct **fAFFlattenAllComponentPaths() {
  if((Paths =
           (AFProduct **)malloc(sizeof(AFProduct*) * AFComponentCounter)) == NULL) {
    printf("#fAF: Not enough memory for AFProduct pointers!");
    exit(EXIT_FAILURE);
  }

  for (uint64_t I = 0; I < AFs->ListLength; ++I) {
    for (int J = 0; J < AFs->AFItems[I]->NumAFComponents; ++J) {
      AFProduct *ProductObjectWalker = &*(AFs->AFItems[I]->Components[J]);
      Paths[ProductObjectWalker->ItemId] = &*ProductObjectWalker;
      for (int K = 1; K < AFs->AFItems[I]->Components[J]->Height; ++K) {
        Paths[ProductObjectWalker->ProductTail->ItemId] = &*(ProductObjectWalker->ProductTail);
        ProductObjectWalker = ProductObjectWalker->ProductTail;
      }
    }
  }
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
    if (fprintf(FP,
                "\t\t{\n"
                "\t\t\t\"ProductItemId\": %d,\n"
                "\t\t\t\"ACItemId\":%d,\n"
                "\t\t\t\"ACItemString\": \"%s\",\n"
                "\t\t\t\"ProductTailItemId\": %d,\n"
                "\t\t\t\"Input\": \"%s\",\n"
                "\t\t\t\"AF\": %lf,\n",
                (*ObjectToStore)->Components[J]->ItemId,
                (*ObjectToStore)->Components[J]->Factor->ItemId,
                (*ObjectToStore)->Components[J]->Factor->ResultVar,
                (*ObjectToStore)->Components[J]->
                        ProductTail!=NULL?(*ObjectToStore)->Components[J]->ProductTail->ItemId:
                    -1,
                (*ObjectToStore)->Components[J]->Input,
                isnan((*ObjectToStore)->Components[J]->AF)?-1.0:
                                                           (*ObjectToStore)->Components[J]->AF) > 0) {
      fprintf(FP, "\t\t\t\"Path(AFProductIds)\": [%d", ProductPath[0]->ItemId);
      for (int K = 1; K < (*ObjectToStore)->Components[J]->Height; ++K)
        fprintf(FP, ", %d", ProductPath[K]->ItemId);

      if (J == (*ObjectToStore)->NumAFComponents-1)
        fprintf(FP, "]\n\t\t}\n");
      else
        fprintf(FP, "]\n\t\t},\n");
    }
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

int fAFComparator(const void *a, const void *b) {
  double AF1 = (*(AFProduct **)a)->AF;
  double AF2 = (*(AFProduct **)b)->AF;

  if(AF2 > AF1)
    return 1;
  else if(AF2 == AF1)
    return 0;
  else
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

// Input: AC Record corresponding to the current instruction and the AF Records
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
AFItem **fAFComputeAF(ACItem **AC, AFItem ***AFItemWRTOperands, int NumOperands) {
#if FAF_DEBUG>=2
  printf("\nCreating AFItem\n");
  printf("\tACId: %d\n", (*AC)->ItemId);
  printf("\tNumber of Operands: %d\n", NumOperands);
  printf("\tRootNode: %s\n", (*AC)->ResultVar);
  printf("\tComponents:\n");
  for (int I = 0; I < NumOperands; ++I) {
    if(AFItemWRTOperands[I] != NULL)
      printf("\t\tAFId %d #Components: %d\n",
             (*AFItemWRTOperands[I])->ItemId,
             (*AFItemWRTOperands[I])->NumAFComponents);
  }

  printf("\n");
#endif

  //  Create a new AF record and initialize data members and allocate memory for
  //    AFComponents array.
  AFItem *NewAFItem = NULL;
  fAFCreateAFItem(&NewAFItem);
  NewAFItem->ItemId = AFItemCounter;
  NewAFItem->NumAFComponents=0;

  int TotalAFComponents = 0;

#if MEMORY_OPT
  TotalAFComponents+=NumOperands;
#else
  // Computing the number of Components/AFPaths that contribute to the Relative
  // Error of this instruction.
  for (int I = 0; I < NumOperands; ++I) {
    if (AFItemWRTOperands[I] != NULL)
      TotalAFComponents+=(*AFItemWRTOperands[I])->NumAFComponents;
    else
      TotalAFComponents++;
  }
#endif
#if FAF_DEBUG>=2
  printf("\tTotalAFComponents: %d\n\n", TotalAFComponents);
#endif

  if((NewAFItem->Components =
           (AFProduct **)malloc(sizeof(AFProduct*)*TotalAFComponents)) == NULL) {
    printf("#fAF: Not enough memory for AFProduct Array!");
    exit(EXIT_FAILURE);
  }

  AFItemCounter++;


  // Generating AFProduct Data
  // Loop over the AF Records corresponding to each operand and for each operand:
  for (int I = 0; I < NumOperands; ++I) {
    if(AFItemWRTOperands[I] != NULL) {
#if MEMORY_OPT
      assert((*AFItemWRTOperands[I])->Components[0] != NULL);
      double GreatestAF = (*AFItemWRTOperands[I])->Components[0]->AF *
                          (*AC)->ACWRTOperands[I];
      int CandidateIndex = 0;
#endif

      // Loop through all Components of this child. Do not skip 0 to optimize
      //  getting the greatest AF or you will miss some components.
      for (int J = 0; J < (*AFItemWRTOperands[I])->NumAFComponents; ++J) {
#if MEMORY_OPT
        // Get the greatest AF and index of the corresponding component.
        if(GreatestAF < (*AFItemWRTOperands[I])->Components[J]->AF *
                             (*AC)->ACWRTOperands[I]) {
          GreatestAF = (*AFItemWRTOperands[I])->Components[J]->AF *
                       (*AC)->ACWRTOperands[I];
          CandidateIndex = J;
        }
#else
        // Create a new AFProduct and copy the corresponding AFProduct from
        // the AFItem for that operand and also the ACItem and set the AF.
        AFProduct *NewAFComponent = NULL;
        fAFCreateAFComponent(&NewAFComponent);
        NewAFComponent->Input = (*AFItemWRTOperands[I])->Components[J]->Input;
        NewAFComponent->ItemId = AFComponentCounter;
        NewAFComponent->Factor = *AC;
        NewAFComponent->ProductTail = (*AFItemWRTOperands[I])->Components[J];
        NewAFComponent->Height = (*AFItemWRTOperands[I])->Components[J]->Height+1;
        NewAFComponent->AF = (*AFItemWRTOperands[I])->Components[J]->AF *
                             (*AC)->ACWRTOperands[I];

        AFComponentCounter++;

        // Add this new AFProduct to the new AFItem
        NewAFItem->Components[NewAFItem->NumAFComponents] = NewAFComponent;
        NewAFItem->NumAFComponents++;
#endif
      }

#if MEMORY_OPT
      // Create a new AFProduct and copy the corresponding AFProduct from
      // the AFItem for that operand and also the ACItem and set the AF.
      AFProduct *NewAFComponent = NULL;
      fAFCreateAFComponent(&NewAFComponent);
      NewAFComponent->Input = (*AFItemWRTOperands[I])->Components[CandidateIndex]->Input;
      NewAFComponent->ItemId = AFComponentCounter;
      NewAFComponent->Factor = *AC;
      NewAFComponent->ProductTail = (*AFItemWRTOperands[I])->Components[CandidateIndex];
      NewAFComponent->Height = (*AFItemWRTOperands[I])->Components[CandidateIndex]->Height+1;
      NewAFComponent->AF = (*AFItemWRTOperands[I])->Components[CandidateIndex]->AF *
                           (*AC)->ACWRTOperands[I];

      AFComponentCounter++;

      // Add this new AFProduct to the new AFItem
      NewAFItem->Components[NewAFItem->NumAFComponents] = NewAFComponent;
      NewAFItem->NumAFComponents++;
#endif
    } else if(strlen((*AC)->OperandNames[I]) != 0) {
      // Create a new AFProduct and copy the ACItem and set the AF.
      AFProduct *NewAFComponent = NULL;
      fAFCreateAFComponent(&NewAFComponent);
      NewAFComponent->Input = (*AC)->OperandNames[I];
      NewAFComponent->ItemId = AFComponentCounter;
      NewAFComponent->Factor = *AC;
      NewAFComponent->ProductTail = NULL;
      NewAFComponent->Height = 1;
      NewAFComponent->AF = (*AC)->ACWRTOperands[I];

      AFComponentCounter++;

      // Add this new AFProduct to the new AFItem
      NewAFItem->Components[NewAFItem->NumAFComponents] = NewAFComponent;
      NewAFItem->NumAFComponents++;
    }
  }

  //  Add AFItem to the AFTable
  AFs->AFItems[AFs->ListLength] = NewAFItem;
  AFs->ListLength++;

#if FAF_DEBUG>=2
  printf("\nAFItem Created\n");
  printf("\tAFId: %d\n", AFs->AFItems[AFs->ListLength-1]->ItemId);
  printf("\tNumComponents: %d\n", AFs->AFItems[AFs->ListLength-1]->NumAFComponents);
  printf("\n");
#endif

  //  Return the AFItem
  return &AFs->AFItems[AFs->ListLength-1];
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
    AFProduct **ProductPath= fAFFlattenAFComponentsPath(AFs->AFItems[AFs->ListLength-1]->Components[I]);
    printf("AF: %0.15lf of Node:%d WRT Input:%s through path: [",
           AFs->AFItems[AFs->ListLength-1]->Components[I]->AF,
           AFs->AFItems[AFs->ListLength-1]->Components[I]->ItemId,
           AFs->AFItems[AFs->ListLength-1]->Components[I]->Input);

    printf("%d", ProductPath[0]->ItemId);
    for (int K = 1; K < AFs->AFItems[AFs->ListLength-1]->Components[I]->Height; ++K)
      printf(", %d", ProductPath[K]->ItemId);
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

  // Printing Results
  printf("\n");
  printf("The top Amplification Paths are:\n");
  for (int I = 0; I < min(10, AFComponentCounter); ++I) {
    AFProduct **ProductPath= fAFFlattenAFComponentsPath(Paths[I]);
    printf("AF: %0.15lf of Node:%d WRT Input:%s through path: [",
           Paths[I]->AF,
           Paths[I]->ItemId,
           Paths[I]->Input);

    printf("%d", ProductPath[0]->ItemId);
    for (int K = 1; K < Paths[I]->Height; ++K)
      printf(", %d", ProductPath[K]->ItemId);
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
