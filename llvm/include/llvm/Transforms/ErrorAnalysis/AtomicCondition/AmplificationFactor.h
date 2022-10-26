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
#define CG_NODES_WORK_LIST_SIZE 100000
#define AF_ITEM_LIST_SIZE 10000
#define MAX_AF_PATH_LENGTH 150
#define MAX_AF_STRING_LENGTH 3000

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

/* ------------------------------- AFComponent -------------------------------*/

struct AFComponent {
  OutputRegister
  ACItem** AFPath;
  double* AF;
};

/* ---------------------------------- AFItem ---------------------------------*/

struct AFItem {
  int ItemId;
  AFComponent** Components;

  uint64_t AFPathLength;

};

typedef struct AFItem AFItem;

void fAFCreateAFItem(AFItem **AddressToAllocateAt) {
  if((*AddressToAllocateAt = (AFItem *)malloc(sizeof(AFItem))) == NULL) {
    printf("#fAF: Not enough memory for AFItem!");
    exit(EXIT_FAILURE);
  }

  return ;
}

// Setting AFItem Values at Location.
void fAFInitializeAFItem(AFItem **Location, CGNode *Node, CGNode *WRTNode) {
  fAFCreateAFItem(Location);

  (*Location)->Node = Node;
  (*Location)->WRTNode = WRTNode;

  // Setting the Amplification Path
  if(((*Location)->AFPathPointer =
           (CGNode **)malloc(sizeof(CGNode*) * MAX_AF_PATH_LENGTH)) == NULL) {
    printf("#fAF: Not enough memory for CGNode pointers!");
    exit(EXIT_FAILURE);
  }

  (*Location)->AFPathLength=0;
  (*Location)->AFPathPointer[(*Location)->AFPathLength] = NULL;
  (*Location)->AF = 1;
  (*Location)->AmplifiedRelativeError = 0;

  if (((*Location)->AFString =
           (char *)malloc(sizeof(char) * MAX_AF_STRING_LENGTH)) == NULL) {
    printf("#fAF: Not enough memory for AFString!");
    exit(EXIT_FAILURE);
  }
  (*Location)->AFString[0] = '\0';

  return ;
}

// Setting AFItem Values at Location.
void fAFAddAFItemCorrespondingChildNode(AFItem **Location, CGNode *Node, CGNode *WRTNode, int WRT,
                  AFItem *ParentAFItem) {
  fAFCreateAFItem(Location);
  (*Location)->Node = Node;
  if(!WRT)
    (*Location)->WRTNode = WRTNode->LeftNode;
  else if(WRT==1)
    (*Location)->WRTNode = WRTNode->RightNode;

  // Setting the Amplification Path
  if(((*Location)->AFPathPointer =
           (CGNode **)malloc(sizeof(CGNode*) * MAX_AF_PATH_LENGTH)) == NULL) {
    printf("#fAF: Not enough memory for CGNode pointers!");
    exit(EXIT_FAILURE);
  }

  fAFCopyCGNodePointerArray(&(*Location)->AFPathPointer,
                            &ParentAFItem->AFPathPointer,
                            ParentAFItem->AFPathLength);

  if(!WRT) {
    (*Location)->AFPathPointer[ParentAFItem->AFPathLength] = WRTNode->LeftNode;
    (*Location)->AF = ParentAFItem->AF *
                      (*WRTNode->ACRecord)->ACWRTX;
  } else if(WRT==1) {
    (*Location)->AFPathPointer[ParentAFItem->AFPathLength] = WRTNode->RightNode;
    (*Location)->AF = ParentAFItem->AF *
                      (*WRTNode->ACRecord)->ACWRTY;
  }

  (*Location)->AFPathLength = ParentAFItem->AFPathLength + 1;

  (*Location)->AmplifiedRelativeError = (*Location)->AF *
                                        fp64OpError[WRTNode->Kind];

  // Allocating memory for the string representation of Amplification
  // factor and storing the string
  if (((*Location)->AFString = (char *)malloc(sizeof(char) * 3000)) == NULL) {
    printf("#fAF: Out of memory error!");
    exit(EXIT_FAILURE);
  }
  if(strlen(ParentAFItem->AFString) != 0)
    strcpy((*Location)->AFString, (ParentAFItem->AFString));
  else
    strcat((*Location)->AFString, "1.0");

  if(!WRT && strlen((*WRTNode->ACRecord)->ACWRTXstring) != 0) {
    strcat((*Location)->AFString, "*");
    strcat((*Location)->AFString,
           (*WRTNode->ACRecord)->ACWRTXstring);
  } else if (WRT==1 && strlen((*WRTNode->ACRecord)->ACWRTYstring) != 0) {
    strcat((*Location)->AFString, "*");
    strcat((*Location)->AFString,
           (*WRTNode->ACRecord)->ACWRTYstring);
  }

  return ;
}

void fAFPrintAFItem(AFItem *Item) {
  printf("\t\tInstruction String: %s\n",
         Item->Node->InstructionString);
  printf("\t\tNode Id: %d\n",
         Item->Node->NodeId);
  printf("\t\tWRTInstructionString: %s\n",
         Item->WRTNode->InstructionString);
  printf("\t\tWRTNodeID: %d\n",
         Item->WRTNode->NodeId);
  printf("\t\tAF Path Length: %lu\n",
         Item->AFPathLength);
  printf("\t\tAF Path: [%d",
         Item->AFPathPointer[0]->NodeId);
  for (int I = 1; (uint64_t)I < Item->AFPathLength;
       ++I) {
    printf(", %d",
           Item->AFPathPointer[I]->NodeId);
  }
  printf("]\n");
  printf("\t\tAF: %0.15lf\n",
         Item->AF);
  printf("\t\tAmplifiedRelativeError: %0.15lf\n",
         Item->AmplifiedRelativeError);
}

/* --------------------------------- AFTable ---------------------------------*/

struct AFTable {
  uint64_t ListLength;
  struct AFItem **AFItems;
};

typedef struct AFTable AFTable;

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/

AFTable *AFs;
AFTable *AnalysisResult;

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

int fAFisMemoryOpInstruction(char *InstructionString) {
  if(strstr(InstructionString, "load")!=NULL ||
      strstr(InstructionString, "alloca")!=NULL)
    return 1;
  return 0;
}

int fAFDoubleAFComparator(const void *a, const void *b) {
  double AF1 = (*(AFItem **)a)->AF;
  double AF2 = (*(AFItem **)b)->AF;

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
  // Allocating Memory to the AF table itself
  if(( AFs = (AFTable*)malloc(sizeof(AFTable))) == NULL) {
    printf("#fAF: Queue out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocating Memory for the AFItem Pointers
  if((AFs->AFItems =
           (AFItem **)malloc(sizeof(AFItem*) * AF_ITEM_LIST_SIZE)) == NULL) {
    printf("#fAF: Not enough memory for AFItem pointers!");
    exit(EXIT_FAILURE);
  }
  AFs->ListLength = 0;

  // Allocating Memory to the Result table itself
  if(( AnalysisResult = (AFTable*)malloc(sizeof(AFTable))) == NULL) {
    printf("#fAF: Queue out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocating Memory for the AFItem Pointers
  if((AnalysisResult->AFItems =
           (AFItem **)malloc(sizeof(AFItem*) * AF_ITEM_LIST_SIZE)) == NULL) {
    printf("#fAF: Not enough memory for AFItem pointers!");
    exit(EXIT_FAILURE);
  }
  AnalysisResult->ListLength = 0;

  return ;
}

/*----------------------------------------------------------------------------*/
/* Analysis Functions                                                         */
/*----------------------------------------------------------------------------*/

// Input: An LLVM instruction that has a computation graph node or is a Phi node.
// Steps:
//  Resolve to a Non-Phi Instruction
//  Get the Node corresponding to the ResolvedInstruction from the Instruction
//  Node map.
//
void fAFAnalysis(char *InstructionToAnalyse) {
  assert(fCGisRegister(InstructionToAnalyse));

#if FAF_DEBUG
  printf("\nPerforming Amplification Factor Calculation\n");
  printf("\tInstruction to Analyse: %s\n", InstructionToAnalyse);
#endif

  // Resolve to a Non-phi Instruction
  char *ResolvedInstruction=fCGperformPHIResolution(InstructionToAnalyse);

  // Search for node corresponding to Instruction String in the InstructionNodeMap
  // and retrieve the Node in the Computation Graph.
  CGNode *RootNode = fCGInstructionNodeMapGet(CG->InstructionNodeMap,
                                               ResolvedInstruction);

#if FAF_DEBUG
  printf("\tFound Root Node: %s\n", RootNode->InstructionString);
#endif

#if FAF_DUMP_SYMBOLIC
  // Create a directory if not present
  char *DirectoryName = (char *)malloc((strlen(LOG_DIRECTORY_NAME)+1) * sizeof(char));
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  char ExecutionId[5000];
  char FileName[5000];
  char NodeIdString[20];
  FileName[0] = '\0';
  strcpy(FileName, strcat(strcpy(FileName, DirectoryName), "/fAF_"));

  fACGenerateExecutionID(ExecutionId);
  strcat(ExecutionId, "_");
  sprintf(NodeIdString, "%d", RootNode->NodeId);
  strcat(ExecutionId, NodeIdString);
  strcat(ExecutionId, ".txt");
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

  // Output file for Relative Error Expression
  FILE *FP = fopen(FileName, "w");

  // Allocating memory for string representation of Relative Error Expression
  char *RelativeErrorString;
  if((RelativeErrorString = (char *)malloc(sizeof(char) * 20000)) == NULL) {
    printf("No memory for relative error string");
    exit(EXIT_FAILURE);
  }
  RelativeErrorString[0] = '\0';
#endif

  // Create a Queue object and add RootNode to front of Queue
  ProcessingQItem *QItem;
  if( (QItem = (ProcessingQItem *)malloc((size_t)((int64_t)sizeof(ProcessingQItem)))) == NULL) {
    printf("#AF: Not enough memory for a ProcessingQItem!");
    exit(EXIT_FAILURE);
  }
  QItem->Node = RootNode;
  QItem->NextNode = NULL;
  ProcessingQ WorkQ = {QItem, QItem, 1};
  ProcessingQItem *WRTNode = QItem;

  // Store Amplification factor of RootNode with Itself in table as 1
  fAFInitializeAFItem(&AFs->AFItems[AFs->ListLength], RootNode,
                      WRTNode->Node);

  AFs->AFItems[AFs->ListLength]->AFPathPointer[AFs->AFItems[AFs->ListLength]->AFPathLength] =
      WRTNode->Node;
  AFs->AFItems[AFs->ListLength]->AFPathLength+=1;

  AFs->ListLength++;

  if (fAFisMemoryOpInstruction(AFs->AFItems[AFs->ListLength-1]->WRTNode->InstructionString)) {
    // Adding this node to the ResultStore
    AnalysisResult->AFItems[AnalysisResult->ListLength] = AFs->AFItems[AFs->ListLength-1];
    AnalysisResult->ListLength++;
    printf("Debugging:\n");
  }

#if FAF_DUMP_SYMBOLIC
  strcat(RelativeErrorString, "(0");
#endif

  // Amplification Factor Calculation
  // Loop till Queue is empty -> QLength == 0
  while(WorkQ.QLength != 0) {
#if FAF_DEBUG>=2
  ProcessingQItem *ProcessingQItemPointer;
  printf("\n\tPrinting the Node Processing Q of size: %lu\n", WorkQ.QLength);
  printf("\tQ Front - Node Id: %d, Instruction String: %s\n",
         WorkQ.Front->Node->NodeId,
         WorkQ.Front->Node->InstructionString);
  printf("\tQ Back - Node Id: %d, Instruction String: %s\n",
         WorkQ.Back->Node->NodeId,
         WorkQ.Back->Node->InstructionString);
  printf("\tQueue:\n");
  for (ProcessingQItemPointer = WorkQ.Front;
       ProcessingQItemPointer!=NULL;
       ProcessingQItemPointer = ProcessingQItemPointer->NextNode) {
    printf("\t\tNode Id: %d, Instruction String: %s\n",
           ProcessingQItemPointer->Node->NodeId,
           ProcessingQItemPointer->Node->InstructionString);
  }
#endif
    // Get the front of Q
    WRTNode = WorkQ.Front;

    // From AF Table, get AF of WRTNode at RootNode
    uint64_t RecordCounter;
    AFItem **AFItemPointer;

#if FAF_DEBUG>=2
    printf("\n\tFinding AF at RootNode %d WRTNode %d\n",
           RootNode->NodeId, WRTNode->Node->NodeId);
#endif
    for(RecordCounter = 0, AFItemPointer = AFs->AFItems;
         RecordCounter < AFs->ListLength && (*AFItemPointer)->WRTNode->NodeId != WRTNode->Node->NodeId;
         RecordCounter++, AFItemPointer++) {
#if FAF_DEBUG>=2
      printf("\t\tAF@Node Id: %d, WRT Node Id: %d is AF=%f\n",
             (*AFItemPointer)->Node->NodeId, (*AFItemPointer)->WRTNode->NodeId, (*AFItemPointer)->AF);
#endif
    }

#if FAF_DEBUG
    printf("\tRequired AF@Node Id: %d, WRT Node Id: %d is AF=%f\n",
           (*AFItemPointer)->Node->NodeId, (*AFItemPointer)->WRTNode->NodeId, (*AFItemPointer)->AF);
#endif

    // Calculate AF and Push nodes to process on WorkQ
    switch (WRTNode->Node->Kind) {
    case 0:
#if FAF_DEBUG
      printf("\tIn Register case\n");
#endif
      break;
    case 1:
#if FAF_DEBUG
      printf("\tIn Unary case\n");
#endif

      // Multiply above AF with left AC and store in table
      if(WRTNode->Node->LeftNode != NULL) {
        fAFAddAFItemCorrespondingChildNode(&AFs->AFItems[AFs->ListLength],
                                           RootNode, WRTNode->Node,
                                           0, *AFItemPointer);

#if FAF_DUMP_SYMBOLIC
        // Appending relative error contribution of node in string format to relative
        // error string.
        strcat(RelativeErrorString, "+");
        char IntroducedErrorString[30];
        IntroducedErrorString[0] = '\0';
        sprintf(IntroducedErrorString, "%0.16lf", fp64OpError[WRTNode->Node->Kind]);
        strcat(RelativeErrorString, IntroducedErrorString);
        if(strlen((*AFItemPointer)->AFString) != 0) {
          strcat(RelativeErrorString, "*");
          strcat(RelativeErrorString, AFs->AFItems[AFs->ListLength]->AFString);
        }
#endif

        AFs->ListLength++;

        if (fAFisMemoryOpInstruction(AFs->AFItems[AFs->ListLength-1]->WRTNode->InstructionString)) {
          // Adding this node to the ResultStore
          AnalysisResult->AFItems[AnalysisResult->ListLength] = AFs->AFItems[AFs->ListLength-1];
          AnalysisResult->ListLength++;
        }

#if FAF_DEBUG>=2
        printf("\n\tNew AFItem Stored:\n");
        fAFPrintAFItem(AFs->AFItems[AFs->ListLength - 1]);
#endif

        fAFAddChildToWorkQ(&WorkQ, WRTNode->Node, 0);
      }
      break;
    case 2:
#if FAF_DEBUG
      printf("\tIn Binary case\n");
#endif

      // Multiply above AF with left AC and store in table
      if(WRTNode->Node->LeftNode != NULL) {
        fAFAddAFItemCorrespondingChildNode(&AFs->AFItems[AFs->ListLength],
                                           RootNode, WRTNode->Node,
                                           0, *AFItemPointer);

#if FAF_DUMP_SYMBOLIC
        // Appending relative error contribution of node in string format to relative
        // error string.
        strcat(RelativeErrorString, "+");
        char IntroducedErrorString[30];
        IntroducedErrorString[0] = '\0';
        sprintf(IntroducedErrorString, "%0.16lf", fp64OpError[WRTNode->Node->Kind]);
        strcat(RelativeErrorString, IntroducedErrorString);
        if(strlen((*AFItemPointer)->AFString) != 0) {
          strcat(RelativeErrorString, "*");
          strcat(RelativeErrorString, (*AFItemPointer)->AFString);
        }
#endif

        AFs->ListLength++;

        if (fAFisMemoryOpInstruction(AFs->AFItems[AFs->ListLength-1]->WRTNode->InstructionString)) {
          // Adding this node to the ResultStore
          AnalysisResult->AFItems[AnalysisResult->ListLength] = AFs->AFItems[AFs->ListLength-1];
          AnalysisResult->ListLength++;
        }

#if FAF_DEBUG>=2
        printf("\n\tNew AFItem Stored:\n");
        fAFPrintAFItem(AFs->AFItems[AFs->ListLength - 1]);
#endif

        fAFAddChildToWorkQ(&WorkQ, WRTNode->Node, 0);
      }

      // Multiply above AF with right AC and store in table
      if(WRTNode->Node->RightNode != NULL) {
        fAFAddAFItemCorrespondingChildNode(&AFs->AFItems[AFs->ListLength],
                                           RootNode, WRTNode->Node,
                                           1, *AFItemPointer);

#if FAF_DUMP_SYMBOLIC
        // Appending relative error contribution of node in string format to relative
        // error string.
        strcat(RelativeErrorString, "+");
        char IntroducedErrorString[30];
        IntroducedErrorString[0] = '\0';
        sprintf(IntroducedErrorString, "%0.16lf", fp64OpError[WRTNode->Node->Kind]);
        strcat(RelativeErrorString, IntroducedErrorString);
        if(strlen((*AFItemPointer)->AFString) != 0) {
          strcat(RelativeErrorString, "*");
          strcat(RelativeErrorString, (*AFItemPointer)->AFString);
        }
#endif

        AFs->ListLength++;

        if (fAFisMemoryOpInstruction(AFs->AFItems[AFs->ListLength-1]->WRTNode->InstructionString)) {
          // Adding this node to the ResultStore
          AnalysisResult->AFItems[AnalysisResult->ListLength] = AFs->AFItems[AFs->ListLength-1];
          AnalysisResult->ListLength++;
        }

#if FAF_DEBUG>=2
        printf("\n\tNew AFItem Stored:\n");
        fAFPrintAFItem(AFs->AFItems[AFs->ListLength - 1]);
#endif

        fAFAddChildToWorkQ(&WorkQ, WRTNode->Node, 1);
      }
      break;
    }

    // Popping off Front of Q
    WorkQ.Front = WorkQ.Front->NextNode;
    WorkQ.QLength--;
    free(WRTNode);
  }

#if FAF_DUMP_SYMBOLIC
  strcat(RelativeErrorString, ")");
#endif

#if FAF_DEBUG
  printf("\nDone Analysing fp64 NodeId: %d, Instruction: %s\n",
         RootNode->NodeId, RootNode->InstructionString);
#endif

#if FAF_DUMP_SYMBOLIC
  fprintf(FP, "%s\n", RelativeErrorString);

  fclose(FP);

//  printf("\nRelative Error written to: %s\n", FileName);
#endif

  return ;
}

void fAFPrintTopAmplificationPaths() {
  // Sorting the two lists according to Amplification Factors
  qsort(AnalysisResult->AFItems, AnalysisResult->ListLength,
        sizeof(AFItem *), fAFDoubleAFComparator);

  // Printing Results
  printf("\n");
  printf("The top Amplification Paths are:\n");
  for (int I = 0; I < min(5, AnalysisResult->ListLength); ++I) {
    printf("AF: %f of Node:%d WRT Node:%d through path: [%d",
           AnalysisResult->AFItems[I]->AF,
           AnalysisResult->AFItems[I]->Node->NodeId,
           AnalysisResult->AFItems[I]->WRTNode->NodeId,
           AnalysisResult->AFItems[I]->AFPathPointer[0]->NodeId);
    for (int J = 1; (uint64_t)J < AnalysisResult->AFItems[I]->AFPathLength; ++J) {
      printf(", %d",
             AnalysisResult->AFItems[I]->AFPathPointer[J]->NodeId);
    }
    printf("]\n");
  }
  printf("\n");
}

void fAFStoreAFs() {
  // Generate a file path + file name string to store the AF Records
  char File[5000];
  fAFGenerateFileString(File, "fAF_", ".json");

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
  FILE *FP = fopen(File, "w");
  fprintf(FP, "{\n");

  long unsigned int RecordsStored = 0;

  RecordsStored = 0;

  fprintf(FP, "\t\"AFs\": [\n");
  int I = 0;
  while ((uint64_t)I < AFs->ListLength) {
    if (fprintf(FP,
                "\t\t{\n"
                "\t\t\t\"Result Instruction\": \"%s\",\n"
                "\t\t\t\"Result Node ID\":%d,\n"
                "\t\t\t\"AF WRT Node\": \"%s\",\n"
                "\t\t\t\"WRT Node ID\":%d,\n"
                "\t\t\t\"AF\": %0.15lf,\n"
                "\t\t\t\"Amplified Relative Error\": %0.15lf,\n"
                "\t\t\t\"AF String\": \"%s\",\n",
                AFs->AFItems[I]->Node->InstructionString,
                AFs->AFItems[I]->Node->NodeId,
                AFs->AFItems[I]->WRTNode->InstructionString,
                AFs->AFItems[I]->WRTNode->NodeId,
                AFs->AFItems[I]->AF,
                AFs->AFItems[I]->AmplifiedRelativeError,
                AFs->AFItems[I]->AFString) > 0) {
      fprintf(FP,
              "\t\t\t\"AF Path\": [%d",
              AFs->AFItems[I]->AFPathPointer[0]->NodeId);
      for (int J = 1; (uint64_t)J < AFs->AFItems[I]->AFPathLength; ++J) {
        fprintf(FP,
                ", %d",
                AFs->AFItems[I]->AFPathPointer[J]->NodeId);
      }
      fprintf(FP, "]\n");
      RecordsStored++;

      if (RecordsStored != AFs->ListLength)
        fprintf(FP, "\t\t},\n");
      else
        fprintf(FP, "\t\t}\n");
    }
    I++;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");

  fclose(FP);

  printf("\nAmplification Factors written to file: %s\n", File);
}

#endif // LLVM_AMPLIFICATIONFACTOR_H
