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
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>


enum Operation {
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
  TruncToFloat
};


// Atomic Condition storage
typedef struct FloatACItem {
  int NodeId;
  const char *XName;
  float X;
  const char *YName;
  float Y;
  enum Operation OP;
  float ACWRTX;
  float ACWRTY;
} FloatACItem;

typedef struct DoubleACItem {
  int NodeId;
  const char *XName;
  double X;
  const char *YName;
  double Y;
  enum Operation OP;
  double ACWRTX;
  double ACWRTY;
} DoubleACItem;

typedef struct ACTable {
  uint64_t Size;
  uint64_t FP32ListSize;
  uint64_t FP64ListSize;

  FloatACItem *FP32ACItems;
  DoubleACItem *FP64ACItems;

} ACTable;

/*----------------------------------------------------------------------------*/
/* Constants                                                                    */
/*----------------------------------------------------------------------------*/
#define LOG_DIRECTORY_NAME ".fAF_logs"

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/

ACTable *StorageTable;
int NodeCounter;

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

void fACCreate() {
  ACTable *AtomicConditionsTable = NULL;
  int64_t Size = 1000;

  // Allocating the table itself
  if(( AtomicConditionsTable = (ACTable*)malloc(sizeof(ACTable))) == NULL) {
    printf("#fAC: Storage table out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocate pointers to the FP32 head nodes
  if( (AtomicConditionsTable->FP32ACItems =
           (struct FloatACItem *)malloc((size_t)((int64_t)sizeof(FloatACItem) * Size))) == NULL) {
    printf("#fAC: FP32: table out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocate pointers to the FP64 head nodes
  if( (AtomicConditionsTable->FP64ACItems =
           (struct DoubleACItem *)malloc((size_t)((int64_t)sizeof(DoubleACItem) * Size))) == NULL) {
    printf("#fAC: FP64: table out of memory error!");
    exit(EXIT_FAILURE);
  }

  for(int I = 0; I<Size; I++) {
    AtomicConditionsTable->FP32ACItems[I].NodeId = -1;
    AtomicConditionsTable->FP64ACItems[I].NodeId = -1;
  }
  AtomicConditionsTable->Size = Size;
  AtomicConditionsTable->FP32ListSize = 0;
  AtomicConditionsTable->FP64ListSize = 0;
  StorageTable = AtomicConditionsTable;

  NodeCounter=0;
}


FloatACItem *fCreateFloatACItem(FloatACItem *NewValue) {
  FloatACItem *NewItem = NULL;

  if((NewItem = (FloatACItem *)malloc(sizeof(FloatACItem))) == NULL) {
    printf("#fAC: AC table out of memory error!");
    exit(EXIT_FAILURE);
  }

  NewItem->NodeId = NewValue->NodeId;
  NewItem->XName = NewValue->XName;
  NewItem->X = NewValue->X;
  NewItem->YName = NewValue->YName;
  NewItem->Y = NewValue->Y;
  NewItem->OP = NewValue->OP;
  NewItem->ACWRTX = NewValue->ACWRTX;
  NewItem->ACWRTY = NewValue->ACWRTY;

  return NewItem;
}

DoubleACItem *fCreateDoubleACItem(DoubleACItem *NewValue) {
  DoubleACItem *NewItem = NULL;

  if((NewItem = (DoubleACItem *)malloc(sizeof(DoubleACItem))) == NULL) {
    printf("#fAC: AC table out of memory error!");
    exit(EXIT_FAILURE);
  }

  NewItem->NodeId = NewValue->NodeId;
  NewItem->XName = NewValue->XName;
  NewItem->X = NewValue->X;
  NewItem->YName = NewValue->YName;
  NewItem->Y = NewValue->Y;
  NewItem->OP = NewValue->OP;
  NewItem->ACWRTX = NewValue->ACWRTX;
  NewItem->ACWRTY = NewValue->ACWRTY;

  return NewItem;
}

int fFloatACItemsEqual(FloatACItem *X, FloatACItem *Y)
{
  if (X->NodeId == Y->NodeId)
    return 1;
  return 0;
}

int fDoubleACItemsEqual(DoubleACItem *X, DoubleACItem *Y)
{
  if (X->NodeId == Y->NodeId)
    return 1;
  return 0;
}



/*----------------------------------------------------------------------------*/
/* Insert a key-value pair into a hash table                                  */
/*----------------------------------------------------------------------------*/

void fACSetFloatItem(ACTable *AtomicConditionsTable, FloatACItem *NewValue)
{
  if (AtomicConditionsTable == NULL)
    return;

  FloatACItem *FoundItem    = NULL;
  FoundItem = &AtomicConditionsTable->FP32ACItems[NewValue->NodeId];

  // There's already a pair
  if(FoundItem != NULL && fFloatACItemsEqual(NewValue, FoundItem)) {
    // Increment values
    FoundItem->XName = NewValue->XName;
    FoundItem->X = NewValue->X;
    FoundItem->YName = NewValue->YName;
    FoundItem->Y = NewValue->Y;
    FoundItem->OP = NewValue->OP;
    FoundItem->ACWRTX = NewValue->ACWRTX;
    FoundItem->ACWRTY = NewValue->ACWRTY;
  } else  { // Nope, could't find it
    FloatACItem *NewItem = NULL;
    NewItem = fCreateFloatACItem(NewValue);
    (AtomicConditionsTable->FP32ListSize)++;

    AtomicConditionsTable->FP32ACItems[NewItem->NodeId] = *NewItem;
  }
}

void fACSetDoubleItem(ACTable *AtomicConditionsTable, DoubleACItem *NewValue)
{
  if (AtomicConditionsTable == NULL)
    return;

  DoubleACItem *FoundItem    = NULL;

  FoundItem = &AtomicConditionsTable->FP64ACItems[NewValue->NodeId];

  // There's already a pair
  if(FoundItem != NULL && fDoubleACItemsEqual(NewValue, FoundItem)) {
    // Increment values
    FoundItem->XName = NewValue->XName;
    FoundItem->X = NewValue->X;
    FoundItem->YName = NewValue->YName;
    FoundItem->Y = NewValue->Y;
    FoundItem->OP = NewValue->OP;
    FoundItem->ACWRTX = NewValue->ACWRTX;
    FoundItem->ACWRTY = NewValue->ACWRTY;
  } else  { // Nope, could't find it
    DoubleACItem *NewItem = NULL;
    NewItem = fCreateDoubleACItem(NewValue);
    (AtomicConditionsTable->FP64ListSize)++;

    AtomicConditionsTable->FP64ACItems[NewItem->NodeId] = *NewItem;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------- Driver Functions ----------------------------
// ---------------------------------------------------------------------------

// Driver function selecting atomic condition function for unary float operation
void fACfp32UnaryDriver(const char *XName, float X, enum Operation OP) {
  FloatACItem Item;
  float AC;

  Item.NodeId = NodeCounter;
  Item.XName = XName;
  Item.X = X;
  Item.YName = "";
  Item.Y = 1;
  Item.OP = OP;

  switch (OP) {
  case 4:
    AC = fabs(X * (cos(X)/sin(X)));
//    printf("AC of sin(x) | x=%f is %f.\n", X, AC);
    break;
  case 5:
    AC = fabs(X * tan(X));
//    printf("AC of cos(x) | x=%f is %f.\n", X, AC);
    break;
  case 6:
    AC = fabs(X / (sin(X)*cos(X)));
//    printf("AC of tan(x) | x=%f is %f.\n", X, AC);
    break;
  case 7:
    AC = fabs(X / (sqrt(1-pow(X,2)) * asin(X)));
//    printf("AC of asin(x) | x=%f is %f.\n", X, AC);
    break;
  case 8:
    AC = fabs(-X / (sqrt(1-pow(X,2)) * acos(X)));
//    printf("AC of acos(x) | x=%f is %f.\n", X, AC);
    break;
  case 9:
    AC = fabs(X / (pow(X,2)+1 * atan(X)));
//    printf("AC of atan(x) | x=%f is %f.\n", X, AC);
    break;
  case 10:
    AC = fabs(X * (cosh(X)/sinh(X)));
//    printf("AC of sinh(x) | x=%f is %f.\n", X, AC);
    break;
  case 11:
    AC = fabs(X * tanh(X));
//    printf("AC of cosh(x) | x=%f is %f.\n", X, AC);
    break;
  case 12:
    AC = fabs(X / (sinh(X)*cosh(X)));
//    printf("AC of tanh(x) | x=%f is %f.\n", X, AC);
    break;
  case 13:
    AC = fabsf(X );
//    printf("AC of exp(x) | x=%f is %f.\n", X, AC);
    break;
  case 14:
    AC = fabs(1/log(X));
//    printf("AC of log(x) | x=%f is %f.\n", X, AC);
    break;
  case 15:
    AC = 0.5;
//    printf("AC of sqrt(x) | x=%f is %f.\n", X, AC);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  Item.ACWRTX = AC;
  fACSetFloatItem(StorageTable, &Item);

  return ;
}

// Driver function selecting atomic condition function for binary float operation
void fACfp32BinaryDriver(const char *XName, float X, const char *YName, float Y, enum Operation OP) {
  FloatACItem Item;
  float ACWRTX;
  float ACWRTY;

  Item.NodeId = NodeCounter;
  Item.XName = XName;
  Item.X = X;
  Item.YName = YName;
  Item.Y = Y;
  Item.OP = OP;

  switch (OP) {
  case 0:
    ACWRTX = fabsf(X / (X+Y));
    ACWRTY = fabsf(Y / (X+Y));
//    printf("AC of x+y | x=%f, y=%f WRT x is %f.\n", X, Y, ACWRTX);
//    printf("AC of x+y | x=%f, y=%f WRT y is %f.\n", X, Y, ACWRTY);
    break;
  case 1:
    ACWRTX = fabsf(X / (X-Y));
    ACWRTY = fabsf(Y / (Y-X));
//    printf("AC of x-y | x=%f, y=%f WRT x is %f.\n", X, Y, ACWRTX);
//    printf("AC of x-y | x=%f, y=%f WRT y is %f.\n", X, Y, ACWRTY);
    break;
  case 2:
    ACWRTX=ACWRTY=1.0;
//    printf("AC of x*y | x=%f, y=%f WRT x is %f.\n", X, Y, ACWRTX);
//    printf("AC of x*y | x=%f, y=%f WRT y is %f.\n", X, Y, ACWRTY);
    break;
  case 3:
    ACWRTX=ACWRTY=1.0;
//    printf("AC of x/y | x=%f, y=%f WRT x is %f.\n", X, Y, ACWRTX);
//    printf("AC of x/y | x=%f, y=%f WRT y is %f.\n", X, Y, ACWRTY);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  Item.ACWRTX = ACWRTX;
  Item.ACWRTY = ACWRTY;
  fACSetFloatItem(StorageTable, &Item);

  return ;
}

// Driver function selecting atomic condition function for unary double operation
void fACfp64UnaryDriver(const char *XName, double X, enum Operation OP) {
  DoubleACItem Item;
  double AC;

  Item.NodeId = NodeCounter;
  Item.XName = XName;
  Item.X = X;
  Item.YName = "";
  Item.Y = 1;
  Item.OP = OP;

  switch (OP) {
  case 4:
    AC = fabs(X * (cos(X)/sin(X)));
//    printf("AC of sin(x) | x=%f is %f.\n", X, AC);
    break;
  case 5:
    AC = fabs(X * tan(X));
//    printf("AC of cos(x) | x=%f is %f.\n", X, AC);
    break;
  case 6:
    AC = fabs(X / (sin(X)*cos(X)));
//    printf("AC of tan(x) | x=%f is %f.\n", X, AC);
    break;
  case 7:
    AC = fabs(X / (sqrt(1-pow(X,2)) * asin(X)));
//    printf("AC of asin(x) | x=%f is %f.\n", X, AC);
    break;
  case 8:
    AC = fabs(-X / (sqrt(1-pow(X,2)) * acos(X)));
//    printf("AC of acos(x) | x=%f is %f.\n", X, AC);
    break;
  case 9:
    AC = fabs(X / (pow(X,2)+1 * atan(X)));
//    printf("AC of atan(x) | x=%f is %f.\n", X, AC);
    break;
  case 10:
    AC = fabs(X * (cosh(X)/sinh(X)));
//    printf("AC of sinh(x) | x=%f is %f.\n", X, AC);
    break;
  case 11:
    AC = fabs(X * tanh(X));
//    printf("AC of cosh(x) | x=%f is %f.\n", X, AC);
    break;
  case 12:
    AC = fabs(X / (sinh(X)*cosh(X)));
//    printf("AC of tanh(x) | x=%f is %f.\n", X, AC);
    break;
  case 13:
    AC = fabs(X);
//    printf("AC of exp(x) | x=%f is %f.\n", X, AC);
    break;
  case 14:
    AC = fabs(1/log(X));
//    printf("AC of log(x) | x=%f is %f.\n", X, AC);
    break;
  case 15:
    AC = 0.5;
//    printf("AC of sqrt(x) | x=%f is %f.\n", X, AC);
    break;
  case 16:
    AC = 1.0;
//    printf("AC of trunc(x, fp32) | x=%f is %f.\n", X, AC);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  Item.ACWRTX = AC;
  fACSetDoubleItem(StorageTable, &Item);

  return ;
}

// Driver function selecting atomic condition function for binary double operation
void fACfp64BinaryDriver(const char *XName, double X, const char *YName, double Y, enum Operation OP) {
  DoubleACItem Item;
  double ACWRTX;
  double ACWRTY;

  Item.NodeId = NodeCounter;
  Item.XName = XName;
  Item.X = X;
  Item.YName = YName;
  Item.Y = Y;
  Item.OP = OP;

  switch (OP) {
  case 0:
    ACWRTX = fabs(X / (X+Y));
    ACWRTY = fabs(Y / (X+Y));
//    printf("AC of x+y | x=%f, y=%f WRT x is %lf.\n", X, Y, ACWRTX);
//    printf("AC of x+y | x=%f, y=%f WRT y is %lf.\n", X, Y, ACWRTY);
    break;
  case 1:
    ACWRTX = fabs(X / (X-Y));
    ACWRTY = fabs(Y / (Y-X));
//    printf("AC of x-y | x=%f, y=%f WRT x is %lf.\n", X, Y, ACWRTX);
//    printf("AC of x-y | x=%f, y=%f WRT y is %lf.\n", X, Y, ACWRTY);
    break;
  case 2:
    ACWRTX=ACWRTY=1.0;
//    printf("AC of x*y | x=%f, y=%f WRT x is %lf.\n", X, Y, ACWRTX);
//    printf("AC of x*y | x=%f, y=%f WRT y is %lf.\n", X, Y, ACWRTY);
    break;
  case 3:
    ACWRTX=ACWRTY=1.0;
//    printf("AC of x/y | x=%f, y=%f WRT x is %lf.\n", X, Y, ACWRTX);
//    printf("AC of x/y | x=%f, y=%f WRT y is %lf.\n", X, Y, ACWRTY);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  Item.ACWRTX = ACWRTX;
  Item.ACWRTY = ACWRTY;
  fACSetDoubleItem(StorageTable, &Item);

  return ;
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

void fACStoreResult() {
#if FAF_DEBUG
  // Create a directory if not present
  char *DirectoryName = (char *)malloc((strlen(LOG_DIRECTORY_NAME)+1) * sizeof(char));
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  char ExecutionId[5000];
  char FileName[5000];
  FileName[0] = '\0';
  strcpy(FileName, strcat(strcpy(FileName, DirectoryName), "/fAC_"));

  fACGenerateExecutionID(ExecutionId);
  strcat(ExecutionId, ".json");

  strcat(FileName, ExecutionId);

  // TODO: Build analysis functions with arguments and print the arguments
  // Get program name and input
//  int str_size = 0;
//  for (int i=0; i < _FPC_PROG_INPUTS; ++i)
//    str_size += strlen(_FPC_PROG_ARGS[i]) + 1;
//  char *prog_input = (char *)malloc((sizeof(char) * str_size) + 1);
//  prog_input[0] = '\0';
//  for (int i=0; i < _FPC_PROG_INPUTS; ++i) {
//    strcat(prog_input, _FPC_PROG_ARGS[i]);
//    strcat(prog_input, " ");
//  }

  // Table Output
  FILE *FP = fopen(FileName, "w");
  fprintf(FP, "{\n");

  long unsigned int RecordsStored = 0;

  fprintf(FP, "\t\"FP32\": [\n");
  int I = 0;
  while ((uint64_t)I < StorageTable->Size) {
    if (StorageTable->FP32ACItems[I].NodeId != -1) {
      if (fprintf(FP,
                  "\t\t{\n"
                  "\t\t\t\"NodeId\":%d,\n"
                  "\t\t\t\"XName\": \"%s\",\n"
                  "\t\t\t\"X\": %0.7f,\n"
                  "\t\t\t\"YName\": \"%s\",\n"
                  "\t\t\t\"Y\": %0.7f,\n"
                  "\t\t\t\"Operation\": %d,\n"
                  "\t\t\t\"ACWRTX\": %0.7f,\n"
                  "\t\t\t\"ACWRTY\": %0.7f\n",
                  StorageTable->FP32ACItems[I].NodeId,
                  StorageTable->FP32ACItems[I].XName,
                  StorageTable->FP32ACItems[I].X,
                  StorageTable->FP32ACItems[I].YName,
                  StorageTable->FP32ACItems[I].Y,
                  StorageTable->FP32ACItems[I].OP,
                  StorageTable->FP32ACItems[I].ACWRTX,
                  StorageTable->FP32ACItems[I].ACWRTY) > 0)
        RecordsStored++;

      if (RecordsStored != StorageTable->FP32ListSize)
        fprintf(FP, "\t\t},\n");
      else
        fprintf(FP, "\t\t}\n");
    }
    I++;
  }
  fprintf(FP, "\t],\n");

  RecordsStored = 0;

  fprintf(FP, "\t\"FP64\": [\n");
  I = 0;
  while ((uint64_t)I < StorageTable->Size) {
    if (StorageTable->FP64ACItems[I].NodeId != -1) {
      if (fprintf(FP,
                  "\t\t{\n"
                  "\t\t\t\"NodeId\":%d,\n"
                  "\t\t\t\"XName\": \"%s\",\n"
                  "\t\t\t\"X\": %0.15f,\n"
                  "\t\t\t\"YName\": \"%s\",\n"
                  "\t\t\t\"Y\": %0.15f,\n"
                  "\t\t\t\"Operation\": %d,\n"
                  "\t\t\t\"ACWRTX\": %0.15f,\n"
                  "\t\t\t\"ACWRTY\": %0.15f\n",
                  StorageTable->FP64ACItems[I].NodeId,
                  StorageTable->FP64ACItems[I].XName,
                  StorageTable->FP64ACItems[I].X,
                  StorageTable->FP64ACItems[I].YName,
                  StorageTable->FP64ACItems[I].Y,
                  StorageTable->FP64ACItems[I].OP,
                  StorageTable->FP64ACItems[I].ACWRTX,
                  StorageTable->FP64ACItems[I].ACWRTY) > 0)
        RecordsStored++;

      if (RecordsStored != StorageTable->FP64ListSize)
        fprintf(FP, "\t\t},\n");
      else
        fprintf(FP, "\t\t}\n");
    }
    I++;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");

  fclose(FP);

  printf("\nAtomic Conditions written to file: %s\n", FileName);
#endif
}


#endif // LLVM_ATOMICCONDITION_H

