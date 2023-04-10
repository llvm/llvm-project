//
// Created by tanmay on 6/12/22.
//

#ifndef LLVM_ATOMICCONDITION_H
#define LLVM_ATOMICCONDITION_H

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <regex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

/*----------------------------------------------------------------------------*/
/* Constants                                                                  */
/*----------------------------------------------------------------------------*/
#define LOG_DIRECTORY_NAME ".fAF_logs"
#define AC_ITEM_LIST_SIZE 1000000


/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/

struct ACTable *ACs;
int ACItemCounter;

/*----------------------------------------------------------------------------*/
/* Data Structures and Types                                                  */
/*----------------------------------------------------------------------------*/

enum Func {
  Add,
  Sub,
  Mul,
  Div,
  Sin,
  Cos,
  Tan,
  ArcSin,
  ArcCos,
  ArcTan,
  Sinh,
  Cosh,
  Tanh,
  Exp,
  Log,
  Sqrt,
  Neg,
  FMA,
};


// Atomic Condition storage

struct ACItem {
  int ItemId;
  enum Func F;
  int NumOperands;
  const char *ResultVar;
  double HPResult;
  char **OperandNames;
  double *OperandValues;
  double *ACWRTOperands;
  char **ACStrings;
  const char *FileName;
  int LineNumber;
};

typedef struct ACItem ACItem;

struct ACTable {
  uint64_t ListLength;
  struct ACItem **ACItems;
};

// Function to write NumObjects from ObjectPointerList into file with descriptor
// FP.
void fACStoreACItems(FILE *FP, ACItem **ObjectPointerList, uint64_t NumObjects) {
  assert(FP != NULL && "File descriptor is NULL.");
  assert(ObjectPointerList != NULL && "ObjectPointerList is NULL");

  long unsigned int RecordsStored = 0;

  fprintf(FP, "{\n");

  fprintf(FP, "\t\"ACs\": [\n");
  int I = 0;
  while (I < AC_ITEM_LIST_SIZE) {
    if (ObjectPointerList[I]!=NULL) {
      fprintf(FP,
              "\t\t{\n"
              "\t\t\t\"ItemId\": %d,\n"
              "\t\t\t\"Function\": %d,\n"
              "\t\t\t\"ResultVar\": \"%s\",\n"
              "\t\t\t\"HPResult\": %0.15lf,\n",
              ObjectPointerList[I]->ItemId,
              ObjectPointerList[I]->F,
              ObjectPointerList[I]->ResultVar,
              ObjectPointerList[I]->HPResult);
      for (int J = 0; J < ObjectPointerList[I]->NumOperands; ++J) {
        fprintf(FP,
                "\t\t\t\"Operand %d Name\": \"%s\",\n"
                "\t\t\t\"Operand %d Value\": %0.15lf,\n",
                J,
                ObjectPointerList[I]->OperandNames[J],
                J,
                ObjectPointerList[I]->OperandValues[J]);
      }
      for (int J = 0; J < ObjectPointerList[I]->NumOperands; ++J) {
        fprintf(FP,
                "\t\t\t\"ACWRTOperand %d\": %0.15lf,\n",
                J,
                ObjectPointerList[I]->ACWRTOperands[J]);
      }
      for (int J = 0; J < ObjectPointerList[I]->NumOperands; ++J) {
        fprintf(FP,
                "\t\t\t\"ACStringWRTOp %d\": \"%s\",\n",
                J,
                ObjectPointerList[I]->ACStrings[J]);
      }
      fprintf(FP,"\t\t\t\"File Name\": \"%s\",\n", ObjectPointerList[I]->FileName);
      fprintf(FP,"\t\t\t\"Line Number\": %d\n", ObjectPointerList[I]->LineNumber);

      RecordsStored++;

      if (RecordsStored != NumObjects)
        fprintf(FP, "\t\t},\n");
      else
        fprintf(FP, "\t\t}\n");
    }
    I++;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");
}

typedef struct ACTable ACTable;

// ---------------------------------------------------------------------------
// ---------------------------- Utility Functions ----------------------------
// ---------------------------------------------------------------------------
int fACIsTernaryOperation(enum Func F) {
  return (F == 17);
}

int fACIsBinaryOperation(enum Func F) {
  return (F == 0 ||
          F == 1 ||
          F == 2 ||
          F == 3);
}

int fACIsUnaryOperation(enum Func F) {
  switch (F) {
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
  case 15:
  case 16:
    return 1;
  default:
    return 0;
  }
}

int fACFuncHasXNumOperands(enum Func F) {
  if (fACIsUnaryOperation(F))
    return 1;
  if (fACIsBinaryOperation(F))
    return 2;
  if (fACIsTernaryOperation(F))
    return 3;
  return 0;
}

void fACAppendDoubleToString(char* String, double DoubleValue) {
  char DoubleBuffer[15];
  snprintf(DoubleBuffer, 15, "%lf", DoubleValue);
  strcat(String, DoubleBuffer);
  return ;
}

char *fACDumpAtomicConditionString(char **OperandNames, double *OperandValues,
                                   enum Func F, int WRT) {
  char *ACstring;
  if((ACstring = (char *)malloc(sizeof(char) * 150)) == NULL) {
    printf("#fAC: Out of memory error!");
    exit(EXIT_FAILURE);
  }
  
  ACstring[0] = '\0';
  switch (F) {
  case 0:
    // Appending Numerator
    if(!WRT) {
      strcat(ACstring, "(");
      if (strlen(OperandNames[0]) != 0)
        strcat(ACstring, OperandNames[0]);
      else
        fACAppendDoubleToString(ACstring, OperandValues[0]);
    }
    else if(WRT==1) {
      if (strlen(OperandNames[1]) != 0) {
        strcat(ACstring, "(");
        strcat(ACstring, OperandNames[1]);
      }
      else
        fACAppendDoubleToString(ACstring, OperandValues[1]);
    }

    strcat(ACstring, "/");
    
    // Appending Denominator
    strcat(ACstring, "(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    
    strcat(ACstring, "+");
    
    if (strlen(OperandNames[1]) != 0)
      strcat(ACstring, OperandNames[1]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[1]);
    strcat(ACstring, ")");
    strcat(ACstring, ")");
    break;
  case 1:
    // Appending Numerator
    if(!WRT) {
      if (strlen(OperandNames[0]) != 0) {
        strcat(ACstring, "(");
        strcat(ACstring, OperandNames[0]);
      }
      else
        fACAppendDoubleToString(ACstring, OperandValues[0]);
    }
    else if(WRT==1) {
      if (strlen(OperandNames[1]) != 0) {
        strcat(ACstring, "(");
        strcat(ACstring, OperandNames[1]);
      }
      else
        fACAppendDoubleToString(ACstring, OperandValues[1]);
    }

    strcat(ACstring, "/");
    
    // Appending Denominator
    strcat(ACstring, "(");
    if(!WRT) {
      if (strlen(OperandNames[0]) != 0)
        strcat(ACstring, OperandNames[0]);
      else
        fACAppendDoubleToString(ACstring, OperandValues[0]);
    }
    else if(WRT==1) {
      if (strlen(OperandNames[1]) != 0)
        strcat(ACstring, OperandNames[1]);
      else
        fACAppendDoubleToString(ACstring, OperandValues[1]);
    }

    strcat(ACstring, "-");

    if(!WRT) {
      if (strlen(OperandNames[1]) != 0)
        strcat(ACstring, OperandNames[1]);
      else
        fACAppendDoubleToString(ACstring, OperandValues[1]);
    }
    else if(WRT==1) {
      if (strlen(OperandNames[0]) != 0)
        strcat(ACstring, OperandNames[0]);
      else
        fACAppendDoubleToString(ACstring, OperandValues[0]);
    }

    strcat(ACstring, ")");
    strcat(ACstring, ")");
    break;
  case 2:
  case 3:
    strcat(ACstring, "(1.0)");
    break;
  case 4:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "*");
    strcat(ACstring, "cos(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")/");
    strcat(ACstring, "sin(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")");

    strcat(ACstring, ")");
    break;
  case 5:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "*");
    strcat(ACstring, "tan(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")");

    strcat(ACstring, ")");
    break;
  case 6:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "/(");
    strcat(ACstring, "sin(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")*");
    strcat(ACstring, "cos(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "))");

    strcat(ACstring, ")");
    break;
  case 7:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "/(");
    strcat(ACstring, "sqrt(1-(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "*");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "))*arcsin(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "))");

    strcat(ACstring, ")");
    break;
  case 8:
    strcat(ACstring, "(-");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "/(");
    strcat(ACstring, "sqrt(1-(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "*");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "))*arccos(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "))");

    strcat(ACstring, ")");
    break;
  case 9:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "/(((");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "*");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")+1)*arctan(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "))");

    strcat(ACstring, ")");
    break;
  case 10:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "*");
    strcat(ACstring, "cosh(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")/");
    strcat(ACstring, "sinh(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")");

    strcat(ACstring, ")");
    break;
  case 11:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "*");
    strcat(ACstring, "tanh(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")");

    strcat(ACstring, ")");
    break;
  case 12:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "/(");
    strcat(ACstring, "sinh(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, ")*");
    strcat(ACstring, "cosh(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);
    strcat(ACstring, "))");

    strcat(ACstring, ")");
    break;
  case 13:
    strcat(ACstring, "(");

    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);

    strcat(ACstring, ")");
    break;
  case 14:
    strcat(ACstring, "(");

    strcat(ACstring, "1/log(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);

    strcat(ACstring, "))");
    break;
  case 15:
    strcat(ACstring, "(0.5)");
    break;
  case 16:
    strcat(ACstring, "(1.0)");
    break;
  case 17:
    // Appending Numerator
    if(WRT==0 || WRT == 1) {
      strcat(ACstring, "(");
      strcat(ACstring, "(");
      if (strlen(OperandNames[0]) != 0)
        strcat(ACstring, OperandNames[0]);
      else
        fACAppendDoubleToString(ACstring, OperandValues[0]);

      strcat(ACstring, "*");

      if (strlen(OperandNames[1]) != 0)
        strcat(ACstring, OperandNames[1]);
      else
        fACAppendDoubleToString(ACstring, OperandValues[1]);
      strcat(ACstring, ")");
    }
    else if(WRT==2) {
      if (strlen(OperandNames[2]) != 0) {
        strcat(ACstring, "(");
        strcat(ACstring, OperandNames[2]);
      }
      else
        fACAppendDoubleToString(ACstring, OperandValues[2]);
    }

    strcat(ACstring, "/");

    // Appending Denominator
    strcat(ACstring, "(");
    strcat(ACstring, "(");
    if (strlen(OperandNames[0]) != 0)
      strcat(ACstring, OperandNames[0]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[0]);

    strcat(ACstring, "*");

    if (strlen(OperandNames[1]) != 0)
      strcat(ACstring, OperandNames[1]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[1]);
    strcat(ACstring, ")");

    strcat(ACstring, "+");

    if (strlen(OperandNames[2]) != 0)
      strcat(ACstring, OperandNames[2]);
    else
      fACAppendDoubleToString(ACstring, OperandValues[2]);
    strcat(ACstring, ")");
    strcat(ACstring, ")");
    break;
  default:
    printf("No such operation %d\n", F);
    exit(1);
  }

  return ACstring;
}

/*----------------------------------------------------------------------------*/
/* File Functions                                                             */
/*----------------------------------------------------------------------------*/

// Create a directory
void fAFcreateLogDirectory(char *DirectoryName) {
  struct stat ST;
  if (stat(DirectoryName, &ST) == -1) {
    // TODO: Check the file mode and whether this one is the right one to use.
    mkdir(DirectoryName, 0775);
  }
}

void fACGenerateExecutionID(char* ExecutionId) {
  //size_t len=256;
  // According to Linux manual:
  // Each element of the hostname must be from 1 to 63 characters long
  // and the entire hostname, including the dots, can be at most 253
  // characters long.
  ExecutionId[0] = '\0';
  if(gethostname(ExecutionId, 256) != 0)
    strcpy(ExecutionId, "node-unknown");

  // Maximum size for PID: we assume 2,000,000,000
  int PID = (int)getpid();
  char PIDStr[11];
  PIDStr[0] = '\0';
  sprintf(PIDStr, "%d", PID);
  strcat(ExecutionId, "_");
  strcat(ExecutionId, PIDStr);
}

// Returns a File String. Also creates a Directory if not present.
// Input:
//  File: char* - Pointer to an empty memory block allocated for the File String.
//                File is the absolute path to the file in question
//  FileNamePrefix: char* - A pointer to the Prefix for the file.
//  Extension: char* - A pointer to the Extension string of the file.
// Output:

void fAFGenerateFileString(char *File,
                           const char *FileNamePrefix,
                           const char *Extension) {
  assert(File != NULL && "Memory not allocated for File String");

  // Create a directory if not present
  const int DirectoryNameLen = sizeof LOG_DIRECTORY_NAME;
  char DirectoryName[DirectoryNameLen];
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  char ExecutionId[5000];
  fACGenerateExecutionID(ExecutionId);

  File[0] = '\0';
  strcat(
      strcat(
          strcat(
              strcat(
                  strcpy(
                      File,
                      DirectoryName),
                  "/"),
              FileNamePrefix),
          ExecutionId),
      Extension);
}

/*----------------------------------------------------------------------------*/
/* Memory Allocators                                                          */
/*----------------------------------------------------------------------------*/

void fACCreate() {
#if FAF_DEBUG
  printf("Initializing Atomic Conditions Module\n");
#endif

  // Allocating memory for the table itself
  if(( ACs = (ACTable*)malloc(sizeof(ACTable))) == NULL) {
    printf("#fAC: Not enough memory for ACTable!");
    exit(EXIT_FAILURE);
  }

  // Allocate memory for the pointers.
  if(( ACs->ACItems =
           (ACItem **)malloc((sizeof(ACItem*) * AC_ITEM_LIST_SIZE))) == NULL) {
    printf("#fAC: Not enough memory for ACItem pointers!");
    exit(EXIT_FAILURE);
  }

  ACs->ListLength = 0;

  ACItemCounter=0;

#if FAF_DEBUG
  printf("Atomic Conditions Module Initialized\n");
#endif
}


void fCreateACItem(ACItem **AddressToAllocateAt) {
  if((*AddressToAllocateAt = (ACItem *)malloc(sizeof(ACItem))) == NULL) {
    printf("#fAC: Not enough memory for ACItem!");
    exit(EXIT_FAILURE);
  }

  ACItemCounter++;
  return ;
}

/*----------------------------------------------------------------------------*/
/* Utility Functions                                                             */
/*----------------------------------------------------------------------------*/
int fACItemsEqual(ACItem *X, ACItem *Y)
{
  if (X->ItemId == Y->ItemId)
    return 1;
  return 0;
}



/*----------------------------------------------------------------------------*/
/* Constructors                                                               */
/*----------------------------------------------------------------------------*/

void fACSetACItem(ACTable *AtomicConditionsTable, ACItem *NewValue)
{
  if (AtomicConditionsTable == NULL)
    return;

  ACItem *FoundItem = NULL;

  FoundItem = AtomicConditionsTable->ACItems[NewValue->ItemId];

  // There's already a pair
  if (FoundItem != NULL && fACItemsEqual(NewValue, FoundItem)) {
    FoundItem->F = NewValue->F;
    FoundItem->NumOperands = fACFuncHasXNumOperands(FoundItem->F);
    FoundItem->ResultVar = NewValue->ResultVar;
    FoundItem->HPResult = NewValue->HPResult;
    for (int I = 0; I < FoundItem->NumOperands; ++I) {
      FoundItem->OperandNames[I] = NewValue->OperandNames[I];
      FoundItem->OperandValues[I] = NewValue->OperandValues[I];
      FoundItem->ACStrings[I] = fACDumpAtomicConditionString(NewValue->OperandNames,
                                                             NewValue->OperandValues,
                                                             NewValue->F,
                                                             I);
    }
    FoundItem->ACWRTOperands = NewValue->ACWRTOperands;
    FoundItem->FileName = NewValue->FileName;
    FoundItem->LineNumber = NewValue->LineNumber;
  } else { // Nope, could't find it
    fCreateACItem(&AtomicConditionsTable->ACItems[NewValue->ItemId]);

    AtomicConditionsTable->ACItems[NewValue->ItemId]->ItemId = NewValue->ItemId;
    AtomicConditionsTable->ACItems[NewValue->ItemId]->F = NewValue->F;
    AtomicConditionsTable->ACItems[NewValue->ItemId]->NumOperands =
        fACFuncHasXNumOperands(NewValue->F);
    AtomicConditionsTable->ACItems[NewValue->ItemId]->ResultVar = NewValue->ResultVar;
    AtomicConditionsTable->ACItems[NewValue->ItemId]->HPResult = NewValue->HPResult;

    if((AtomicConditionsTable->ACItems[NewValue->ItemId]->OperandNames =
             (char **)malloc(sizeof(char *) * AtomicConditionsTable->ACItems[NewValue->ItemId]->NumOperands)) == NULL) {
      printf("#fAC: Not enough memory for OperandNames!");
      exit(EXIT_FAILURE);
    }
    if((AtomicConditionsTable->ACItems[NewValue->ItemId]->OperandValues =
             (double *)malloc(sizeof(double) * AtomicConditionsTable->ACItems[NewValue->ItemId]->NumOperands)) == NULL) {
      printf("#fAC: Not enough memory for OperandValues!");
      exit(EXIT_FAILURE);
    }
    for (int I = 0; I < AtomicConditionsTable->ACItems[NewValue->ItemId]->NumOperands; ++I) {
      AtomicConditionsTable->ACItems[NewValue->ItemId]->OperandNames[I] =
          NewValue->OperandNames[I];
      AtomicConditionsTable->ACItems[NewValue->ItemId]->OperandValues[I] =
          NewValue->OperandValues[I];
    }

    AtomicConditionsTable->ACItems[NewValue->ItemId]->ACWRTOperands = NewValue->ACWRTOperands;
    AtomicConditionsTable->ACItems[NewValue->ItemId]->ACStrings = NewValue->ACStrings;
    AtomicConditionsTable->ACItems[NewValue->ItemId]->FileName = NewValue->FileName;
    AtomicConditionsTable->ACItems[NewValue->ItemId]->LineNumber = NewValue->LineNumber;

    AtomicConditionsTable->ListLength++;
  }

  return ;
}

// ---------------------------------------------------------------------------
// ---------------------------- Driver Functions ----------------------------
// ---------------------------------------------------------------------------

ACItem **fACComputeAC(const char *ResultVar,
                      char **OperandNames,
                      double *OperandValues,
                      enum Func F,
                      const char *FileName,
                      int LineNumber) {
  int NumOperands = fACFuncHasXNumOperands(F);

#if FAF_DEBUG >= 2
  printf("\tCreating ACItem Record:\n");
  printf("\t\tFunction       : %d\n", F);
  printf("\t\tResultVar      : %s\n", ResultVar);
  for (int I = 0; I < NumOperands; ++I) {
    printf("\t\tOperand %d Name : %s\n"
           "\t\tOperand %d Value: %0.15lf\n",
           I, OperandNames[I], I, OperandValues[I]);
  }
  printf("\t\tFileName       : %s\n", FileName);
  printf("\t\tLine Number    : %d\n", LineNumber);
  printf("\n");
#endif

  ACItem Item;
  Item.ItemId = ACItemCounter;
  Item.F = F;
  Item.NumOperands = NumOperands;
  Item.ResultVar = ResultVar;
  Item.OperandNames = OperandNames;
  Item.OperandValues = OperandValues;

  if((Item.ACWRTOperands = (double *)malloc(sizeof(double) * Item.NumOperands)) == NULL) {
    printf("#fAC: Not enough memory for ACs!");
    exit(EXIT_FAILURE);
  }
  if((Item.ACStrings = (char **)malloc(sizeof(char*) * Item.NumOperands)) == NULL) {
    printf("#fAC: Not enough memory for ACs!");
    exit(EXIT_FAILURE);
  }

  switch (F) {
  case 0:
    Item.HPResult = OperandValues[0] + OperandValues[1];

    Item.ACWRTOperands[0] = fabs(OperandValues[0] / (OperandValues[0]+OperandValues[1]));
    Item.ACWRTOperands[1] = fabs(OperandValues[1] / (OperandValues[0]+OperandValues[1]));
    //    printf("AC of x+y | x=%lf, y=%lf WRT x is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[0]);
    //    printf("AC of x+y | x=%lf, y=%lf WRT y is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[1]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    Item.ACStrings[1] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 1);
    break;
  case 1:
    Item.HPResult = OperandValues[0] - OperandValues[1];

    Item.ACWRTOperands[0] = fabs(OperandValues[0] / (OperandValues[0]-OperandValues[1]));
    Item.ACWRTOperands[1] = fabs(OperandValues[1] / (OperandValues[1]-OperandValues[0]));
    //    printf("AC of x-y | x=%lf, y=%lf WRT x is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[0]);
    //    printf("AC of x-y | x=%lf, y=%lf WRT y is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[1]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    Item.ACStrings[1] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 1);
    break;
  case 2:
    Item.HPResult = OperandValues[0] * OperandValues[1];

    Item.ACWRTOperands[0]=Item.ACWRTOperands[1]=1.0;
    //    printf("AC of x*y | x=%lf, y=%lf WRT x is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[0]);
    //    printf("AC of x*y | x=%lf, y=%lf WRT y is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[1]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    Item.ACStrings[1] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 1);
    break;
  case 3:
    Item.HPResult = OperandValues[0] / OperandValues[1];

    Item.ACWRTOperands[0]=Item.ACWRTOperands[1]=1.0;
    //    printf("AC of x/y | x=%lf, y=%lf WRT x is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[0]);
    //    printf("AC of x/y | x=%lf, y=%lf WRT y is %lf.\n", OperandValues[0], OperandValues[1], Item.ACWRTOperands[1]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    Item.ACStrings[1] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 1);
    break;
  case 4:
    Item.HPResult = sin(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] * (cos(OperandValues[0])/sin(OperandValues[0])));
    //    printf("AC of sin(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 5:
    Item.HPResult = cos(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] * tan(OperandValues[0]));
    //    printf("AC of cos(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 6:
    Item.HPResult = tan(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] / (sin(OperandValues[0])*cos(OperandValues[0])));
    //    printf("AC of tan(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);
    break;
  case 7:
    assert(OperandValues[0] >= -1.0 && OperandValues[0] <= 1.0 &&
           "asin(x) is not defined for x outside [-1,1]." );
    Item.HPResult = asin(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] / (sqrt(1-pow(OperandValues[0],2)) * asin(OperandValues[0])));
    //    printf("AC of asin(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 8:
    assert(OperandValues[0] >= -1.0 && OperandValues[0] <= 1.0 &&
               "acos(x) is not defined for x outside [-1,1]." );
    Item.HPResult = acos(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(-OperandValues[0] / (sqrt(1-pow(OperandValues[0],2)) * acos(OperandValues[0])));
    //    printf("AC of acos(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 9:
    assert(OperandValues[0] >= -1.0 && OperandValues[0] <= 1.0 &&
               "atan(x) is not defined for x outside [-1,1]." );
    Item.HPResult = atan(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] / (pow(OperandValues[0],2)+1 * atan(OperandValues[0])));
    //    printf("AC of atan(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 10:
    Item.HPResult = sinh(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] * (cosh(OperandValues[0])/sinh(OperandValues[0])));
    //    printf("AC of sinh(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 11:
    Item.HPResult = cosh(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] * tanh(OperandValues[0]));
    //    printf("AC of cosh(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 12:
    Item.HPResult = tanh(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0] / (sinh(OperandValues[0])*cosh(OperandValues[0])));
    //    printf("AC of tanh(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 13:
    Item.HPResult = exp(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(OperandValues[0]);
    //    printf("AC of exp(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 14:
    assert(OperandValues[0] >= 0.0 && "log(x) is not defined for x <= 0." );
    Item.HPResult = log(OperandValues[0]);

    Item.ACWRTOperands[0] = fabs(1/log(OperandValues[0]));
    //    printf("AC of log(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);break;
  case 15:
    assert(OperandValues[0] >= 0.0 && "sqrt(x) is not defined for x <= 0.");
    Item.HPResult = sqrt(OperandValues[0]);

    Item.ACWRTOperands[0] = 0.5;
    //    printf("AC of sqrt(x) | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 16:
    Item.HPResult = -OperandValues[0];

    Item.ACWRTOperands[0] = 1.0;
    //    printf("AC of -x | x=%lf is %lf.\n", OperandValues[0], Item.ACWRTOperands[0]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    break;
  case 17:
    Item.HPResult = OperandValues[0] * OperandValues[1] + OperandValues[2];

    Item.ACWRTOperands[0] = fabs(OperandValues[0]*OperandValues[1] / ((OperandValues[0]*OperandValues[1])+OperandValues[2]));
    Item.ACWRTOperands[1] = fabs(OperandValues[0]*OperandValues[1] / ((OperandValues[0]*OperandValues[1])+OperandValues[2]));
    Item.ACWRTOperands[2] = fabs(OperandValues[2] / ((OperandValues[0]*OperandValues[1])+OperandValues[2]));
    //    printf("AC of xy+z | x=%lf, y=%lf z=%lf WRT x is %lf.\n", OperandValues[0], OperandValues[1], OperandValues[2], Item.ACWRTOperands[0]);
    //    printf("AC of xy+z | x=%lf, y=%lf z=%lf WRT y is %lf.\n", OperandValues[0], OperandValues[1], OperandValues[2], Item.ACWRTOperands[1]);
    //    printf("AC of xy+z | x=%lf, y=%lf z=%lf WRT z is %lf.\n", OperandValues[0], OperandValues[1], OperandValues[2], Item.ACWRTOperands[2]);

    Item.ACStrings[0] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 0);
    Item.ACStrings[1] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 1);
    Item.ACStrings[2] = fACDumpAtomicConditionString(OperandNames, OperandValues, F, 2);
    break;
  default:
    printf("No such operation\n");
    break;
  }
  Item.FileName = FileName;
  Item.LineNumber = LineNumber;

  fACSetACItem(ACs, &Item);


#if FAF_DEBUG >= 2
  NumOperands = ACs->ACItems[Item.ItemId]->NumOperands;

  printf("\tACItem Created:\n");
  printf("\t\tItem Id        : %d\n", ACs->ACItems[Item.ItemId]->ItemId);
  printf("\t\tFunction       : %d\n", ACs->ACItems[Item.ItemId]->F);
  printf("\t\tResultVar      : %s\n", ACs->ACItems[Item.ItemId]->ResultVar);
  printf("\t\tHP Result      : %0.15lf\n", ACs->ACItems[Item.ItemId]->HPResult);
  for (int I = 0; I < NumOperands; ++I) {
    printf("\t\tOperand %d Name : %s\n"
           "\t\tOperand %d Value: %lf\n",
           I, ACs->ACItems[Item.ItemId]->OperandNames[I], I,
           ACs->ACItems[Item.ItemId]->OperandValues[I]);
  }
  for (int I = 0; I < NumOperands; ++I) {
    printf("\t\tACWRTOperand %d : %0.15lf\n", I,
           ACs->ACItems[Item.ItemId]->ACWRTOperands[I]);
  }
  for (int I = 0; I < NumOperands; ++I) {
    printf("\t\tACStringWRTOp %d: %s\n", I,
           ACs->ACItems[Item.ItemId]->ACStrings[I]);
  }

  printf("\t\tFile Name      : %s\n", ACs->ACItems[Item.ItemId]->FileName);
  printf("\t\tLine Number    : %d\n\n", ACs->ACItems[Item.ItemId]->LineNumber);
#endif

  return &ACs->ACItems[Item.ItemId];
}

void fACStoreACs() {
//#if NO_DATA_DUMP
//#else
  printf("\nWriting Atomic Conditions to file.\n");
  // Generate a file path + file name string to store the AC Records
  char File[5000];
  fAFGenerateFileString(File, "fAC_", ".json");

  // Table Output
  FILE *FP;
  if((FP = fopen(File, "w")) != NULL) {
    fACStoreACItems(FP, ACs->ACItems, ACs->ListLength);
    fclose(FP);
  } else {
    printf("%s cannot be opened.\n", File);
  }

  printf("Atomic Conditions written to file: %s\n", File);
//#endif
}


#endif // LLVM_ATOMICCONDITION_H

