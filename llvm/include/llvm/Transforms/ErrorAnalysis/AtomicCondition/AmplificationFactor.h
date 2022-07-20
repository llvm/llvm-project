//
// Created by tanmay on 7/11/22.
//

#ifndef LLVM_AMPLIFICATIONFACTOR_H
#define LLVM_AMPLIFICATIONFACTOR_H

#include "ComputationGraph.h"

typedef struct NodeProcessingQItem {
  CGNode *Node;
  struct NodeProcessingQItem *NextItem;
} NodeProcessingQItem;

typedef struct NodeProcessingQ {
  NodeProcessingQItem *Front;
  NodeProcessingQItem *Back;
  uint64_t Size;
} NodeProcessingQ;

typedef struct FloatAFItem {
//  char *ResultRegName;

  char *ResultInstructionString;
  int ResultNodeID;
//  char *WRTRegName;
  char *WRTInstructionString;
  int WRTNodeID;
  float AF;
} FloatAFItem;

typedef struct DoubleAFItem {
//  char *ResultRegName;
  char *ResultInstructionString;
  int ResultNodeID;
  char *WRTInstructionString;
  int WRTNodeID;
  float AF;
} DoubleAFItem;

typedef struct AnalysisResult {
//  uint64_t PoisonValue;
  FloatAFItem *FloatAFItemPointer;
  uint64_t FloatAFRecords;
  DoubleAFItem *DoubleAFItemPointer;
  uint64_t DoubleAFRecords;
} AnalysisResult;

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/
AnalysisResult *AFResult;
uint64_t Q_SIZE = 1000;

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

void fAFInitialize() {
  AFResult = NULL;
  int64_t Size = 1000;

  // Allocating Memory to the analysis object itself
  if( (AFResult = (AnalysisResult *)malloc(sizeof(AnalysisResult))) == NULL) {
    printf("#AF: Queue out of memory error!");
    exit(EXIT_FAILURE);
  }
//  AFResult->ProcessingQ = NULL;
//  AFResult->PoisonValue = 0xFA84ACA5843FBDB8;
  AFResult->FloatAFItemPointer = NULL;
  AFResult->DoubleAFItemPointer = NULL;
  AFResult->FloatAFRecords = 0;
  AFResult->DoubleAFRecords = 0;

  // Allocating Memory for Amplification Factor Tables
  if( (AFResult->FloatAFItemPointer = (FloatAFItem *)malloc((size_t)((int64_t)sizeof(FloatAFItem) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  if( (AFResult->DoubleAFItemPointer = (DoubleAFItem *)malloc((size_t)((int64_t)sizeof(DoubleAFItem) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

#if FAF_DEBUG
  // Create a directory if not present
  char *DirectoryName = (char *)malloc(strlen(LOG_DIRECTORY_NAME) * sizeof(char));
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);

  char ExecutionId[5000];
  char AFFileName[5000];
  AFFileName[0] = '\0';
  strcpy(AFFileName, strcat(strcpy(AFFileName, DirectoryName), "/program_log_"));

  fACGenerateExecutionID(ExecutionId);
  strcat(ExecutionId, ".txt");

  strcat(AFFileName, ExecutionId);
  printf("Amplification Factors Storage File:%s\n", AFFileName);
#endif
}

void fAFfp32Analysis(char *InstructionToAnalyse) {
  assert(fCGisRegister(InstructionToAnalyse));
#if FAF_DEBUG
  printf("\nPerforming fp32 Amplification Factor Calculation\n");
  printf("\tInstruction to Analyse: %s\n", InstructionToAnalyse);
#endif

  char *ResolvedInstruction=InstructionToAnalyse;
  if(fCGisPHIInstruction(ResolvedInstruction)) {
    // Resolve to a Non-Phi Instruction
    ResolvedInstruction = fCGperformPHIResolution(ResolvedInstruction);
    assert(fCGisRegister(ResolvedInstruction));
  }

  // Search for node corresponding to Instruction String in the InstructionNode
  // Map and retrieve the NodeID as well as the Node from the Computation Graph.
  InstructionNodePair *CurrInstructionNode;
  CGNode *NodeToAnalyse, *CurrCGNode;
  int NodeID;

#if FAF_DEBUG==2
  printf("\tSearching for %s in Instruction -> NodeID Map\n", ResolvedInstruction);
#endif
  for (CurrInstructionNode = CG->InstructionNodeMapHead;
       CurrInstructionNode != NULL &&
       strcmp(CurrInstructionNode->InstructionString, ResolvedInstruction) != 0;
       CurrInstructionNode=CurrInstructionNode->Next){
#if FAF_DEBUG==2
    printf("\t\t%s -> %d\n", CurrInstructionNode->InstructionString, CurrInstructionNode->NodeId);
#endif
  }
  assert(CurrInstructionNode != NULL);
  NodeID = CurrInstructionNode->NodeId;
#if FAF_DEBUG
  printf("\tFound %s -> %d\n", CurrInstructionNode->InstructionString, NodeID);
#endif

#if FAF_DEBUG==2
  printf("\n\tSearching for %d in Nodes List\n", NodeID);
#endif
  for(CurrCGNode = CG->NodesLinkedListHead;
       CurrCGNode != NULL && CurrCGNode->NodeId != NodeID;
       CurrCGNode=CurrCGNode->Next) {
#if FAF_DEBUG==2
    printf("\t\tInstruction String: %s, NodeID: %d\n", CurrCGNode->InstructionString, CurrCGNode->NodeId);
#endif
  }
  assert(CurrCGNode != NULL);
  NodeToAnalyse = CurrCGNode;
#if FAF_DEBUG
  printf("\tFound NodeToAnalyse: %s\n", NodeToAnalyse->InstructionString);
#endif

  // Create a Queue object and add NodeToAnalyse to front of Queue
  NodeProcessingQItem *QItem;
  if( (QItem = (NodeProcessingQItem *)malloc((size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
    printf("#AF: Out of memory error!");
    exit(EXIT_FAILURE);
  }
  QItem->Node = NodeToAnalyse;
  QItem->NextItem = NULL;
  struct NodeProcessingQ ProcessingQ = {QItem, QItem, 1};
  NodeProcessingQItem *WRTNode = QItem;

  // Store Amplification factor of NodeToAnalyse with Itself in table as 1
  AFResult->FloatAFItemPointer->ResultInstructionString = NodeToAnalyse->InstructionString;
  AFResult->FloatAFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
  AFResult->FloatAFItemPointer->WRTInstructionString = WRTNode->Node->InstructionString;
  AFResult->FloatAFItemPointer->WRTNodeID = WRTNode->Node->NodeId;
  AFResult->FloatAFItemPointer->AF = 1;
  AFResult->FloatAFRecords++;

  // Amplification Factor Calculation
  // Loop till Queue is empty -> Front == Back
  while(ProcessingQ.Size != 0 && ProcessingQ.Front != NULL && ProcessingQ.Back != NULL) {
#if FAF_DEBUG==2
    NodeProcessingQItem *ProcessingQItemPointer;
    printf("\n\tPrinting the Node Processing Q of size: %lu\n", ProcessingQ.Size);
    printf("\tQ Front - Node Id: %d, Instruction String: %s\n",
           ProcessingQ.Front->Node->NodeId,
           ProcessingQ.Front->Node->InstructionString);
    printf("\tQ Back - Node Id: %d, Instruction String: %s\n",
           ProcessingQ.Back->Node->NodeId,
           ProcessingQ.Back->Node->InstructionString);
    for (ProcessingQItemPointer = ProcessingQ.Front;
         ProcessingQItemPointer!=NULL;
         ProcessingQItemPointer = ProcessingQItemPointer->NextItem) {
      printf("\t\tNode Id: %d, Instruction String: %s\n",
             ProcessingQItemPointer->Node->NodeId,
             ProcessingQItemPointer->Node->InstructionString);
    }
#endif
    // Get the front of Q
    WRTNode = ProcessingQ.Front;

    // From AF Table, get AF at NodeToAnalyse for WRTNode
    FloatAFItem *AFItemPointer;
    uint64_t RecordCounter;

#if FAF_DEBUG==2
    printf("\n\tFinding AF at NodeToAnalyse %d WRTNode %d\n",
           NodeToAnalyse->NodeId, WRTNode->Node->NodeId);
#endif
    for(RecordCounter = 0, AFItemPointer = AFResult->FloatAFItemPointer;
         RecordCounter < AFResult->FloatAFRecords && AFItemPointer->WRTNodeID != WRTNode->Node->NodeId;
         RecordCounter++, AFItemPointer++) {
#if FAF_DEBUG==2
      printf("\t\tAF@Node Id: %d, WRT Node Id: %d is AF=%f\n",
             AFItemPointer->ResultNodeID, AFItemPointer->WRTNodeID, AFItemPointer->AF);
#endif
    }

#if FAF_DEBUG
    printf("\tRequired AF@Node Id: %d, WRT Node Id: %d is AF=%f\n",
           AFItemPointer->ResultNodeID, AFItemPointer->WRTNodeID, AFItemPointer->AF);
#endif

    float AFofWRTNode = AFItemPointer->AF;

    // Calculate AF and Push nodes to process on ProcessingQ
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
        AFItemPointer = &AFResult->FloatAFItemPointer[AFResult->FloatAFRecords];

        AFItemPointer->ResultInstructionString =
            NodeToAnalyse->InstructionString;
        AFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
        AFItemPointer->WRTInstructionString =
            WRTNode->Node->LeftNode->InstructionString;
        AFItemPointer->WRTNodeID = WRTNode->Node->LeftNode->NodeId;
        AFItemPointer->AF =
            AFofWRTNode *
            StorageTable->FP32ACItems[WRTNode->Node->NodeId].ACWRTX;

        AFResult->FloatAFRecords++;

#if FAF_DEBUG == 2
        printf("\n\tNew AFItem Stored:\n");
        printf("\t\tInstruction String: %s\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .ResultInstructionString);
        printf("\t\tNode Id: %d\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .ResultNodeID);
        printf("\t\tWRTInstructionString: %s\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .WRTInstructionString);
        printf("\t\tWRTNodeID: %d\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .WRTNodeID);
        printf("\t\tAF: %f\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1].AF);
#endif
        if (WRTNode->Node->LeftNode->Kind != 0) {
          // Pushing Node on Processing Q
          if ((QItem = (NodeProcessingQItem *)malloc(
                   (size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
            printf("#AF: Out of memory error!");
            exit(EXIT_FAILURE);
          }
          QItem->Node = WRTNode->Node->LeftNode;
          QItem->NextItem = NULL;
          ProcessingQ.Back->NextItem = QItem;
          ProcessingQ.Back = ProcessingQ.Back->NextItem;

          ProcessingQ.Size++;
        }
      }
      break;
    case 2:
#if FAF_DEBUG
      printf("\tIn Binary case\n");
#endif

      // Multiply above AF with left AC and store in table
      if(WRTNode->Node->LeftNode != NULL) {
        AFItemPointer = &AFResult->FloatAFItemPointer[AFResult->FloatAFRecords];
        AFItemPointer->ResultInstructionString =
            NodeToAnalyse->InstructionString;
        AFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
        AFItemPointer->WRTInstructionString =
            WRTNode->Node->LeftNode->InstructionString;
        AFItemPointer->WRTNodeID = WRTNode->Node->LeftNode->NodeId;
        AFItemPointer->AF =
            AFofWRTNode *
            StorageTable->FP32ACItems[WRTNode->Node->NodeId].ACWRTX;
        AFResult->FloatAFRecords++;

#if FAF_DEBUG == 2
        printf("\n\tNew AFItem Stored:\n");
        printf("\t\tInstruction String: %s\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .ResultInstructionString);
        printf("\t\tNode Id: %d\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .ResultNodeID);
        printf("\t\tWRTInstructionString: %s\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .WRTInstructionString);
        printf("\t\tWRTNodeID: %d\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .WRTNodeID);
        printf("\t\tAF: %f\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1].AF);
#endif

        if (WRTNode->Node->LeftNode->Kind != 0) {
          // Pushing Left Node on Processing Q
          if ((QItem = (NodeProcessingQItem *)malloc(
                   (size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
            printf("#AF: Out of memory error!");
            exit(EXIT_FAILURE);
          }
          QItem->Node = WRTNode->Node->LeftNode;
          QItem->NextItem = NULL;
          ProcessingQ.Back->NextItem = QItem;
          ProcessingQ.Back = ProcessingQ.Back->NextItem;

          ProcessingQ.Size++;
        }
      }

      // Multiply above AF with right AC and store in table
      if(WRTNode->Node->RightNode != NULL) {
        AFItemPointer = &AFResult->FloatAFItemPointer[AFResult->FloatAFRecords];

        AFItemPointer->ResultInstructionString =
            NodeToAnalyse->InstructionString;
        AFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
        AFItemPointer->WRTInstructionString =
            WRTNode->Node->RightNode->InstructionString;
        AFItemPointer->WRTNodeID = WRTNode->Node->RightNode->NodeId;
        AFItemPointer->AF =
            AFofWRTNode *
            StorageTable->FP32ACItems[WRTNode->Node->NodeId].ACWRTY;

        AFResult->FloatAFRecords++;

#if FAF_DEBUG == 2
        printf("\n\tNew AFItem Stored:\n");
        printf("\t\tInstruction String: %s\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .ResultInstructionString);
        printf("\t\tNode Id: %d\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .ResultNodeID);
        printf("\t\tWRTInstructionString: %s\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .WRTInstructionString);
        printf("\t\tWRTNodeID: %d\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1]
                   .WRTNodeID);
        printf("\t\tAF: %f\n",
               AFResult->FloatAFItemPointer[AFResult->FloatAFRecords - 1].AF);
#endif

        if (WRTNode->Node->RightNode->Kind != 0) {
          // Pushing Right Node on Processing Q
          if ((QItem = (NodeProcessingQItem *)malloc(
                   (size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
            printf("#AF: Out of memory error!");
            exit(EXIT_FAILURE);
          }
          QItem->Node = WRTNode->Node->RightNode;
          QItem->NextItem = NULL;
          ProcessingQ.Back->NextItem = QItem;
          ProcessingQ.Back = ProcessingQ.Back->NextItem;

          ProcessingQ.Size++;
        }
      }

      // Popping off Front of Q
      ProcessingQ.Front = ProcessingQ.Front->NextItem;
      ProcessingQ.Size--;
      break;
    }
    free(WRTNode);
  }

#if FAF_DEBUG
  printf("\nDone Analysing fp32 NodeId: %d, Instruction: %s\n",
         NodeToAnalyse->NodeId, NodeToAnalyse->InstructionString);
#endif

  return ;
}


void fAFfp64Analysis(char *InstructionToAnalyse) {
  assert(fCGisRegister(InstructionToAnalyse));
#if FAF_DEBUG
  printf("\nPerforming fp64 Amplification Factor Calculation\n");
  printf("\tInstruction to Analyse: %s\n", InstructionToAnalyse);
#endif

  char *ResolvedInstruction=InstructionToAnalyse;
  if(fCGisPHIInstruction(ResolvedInstruction)) {
    // Resolve to a Non-Phi Instruction
    ResolvedInstruction = fCGperformPHIResolution(ResolvedInstruction);
    assert(fCGisRegister(ResolvedInstruction));
  }

  // Search for node corresponding to Instruction String in the InstructionNode
  // Map and retrieve the NodeID as well as the Node from the Computation Graph.
  InstructionNodePair *CurrInstructionNode;
  CGNode *NodeToAnalyse, *CurrCGNode;
  int NodeID;

#if FAF_DEBUG==2
  printf("\tSearching for %s in Instruction -> NodeID Map\n", ResolvedInstruction);
#endif
  for (CurrInstructionNode = CG->InstructionNodeMapHead;
       CurrInstructionNode != NULL &&
       strcmp(CurrInstructionNode->InstructionString, ResolvedInstruction) != 0;
       CurrInstructionNode=CurrInstructionNode->Next){
#if FAF_DEBUG==2
    printf("\t\t%s -> %d\n", CurrInstructionNode->InstructionString, CurrInstructionNode->NodeId);
#endif
  }
  assert(CurrInstructionNode != NULL);
  NodeID = CurrInstructionNode->NodeId;
#if FAF_DEBUG
  printf("\tFound %s -> %d\n", CurrInstructionNode->InstructionString, NodeID);
#endif

#if FAF_DEBUG==2
  printf("\n\tSearching for %d in Nodes List\n", NodeID);
#endif
  for(CurrCGNode = CG->NodesLinkedListHead;
       CurrCGNode != NULL && CurrCGNode->NodeId != NodeID;
       CurrCGNode=CurrCGNode->Next) {
#if FAF_DEBUG==2
    printf("\t\tInstruction String: %s, NodeID: %d\n", CurrCGNode->InstructionString, CurrCGNode->NodeId);
#endif
  }
  assert(CurrCGNode != NULL);
  NodeToAnalyse = CurrCGNode;
#if FAF_DEBUG
  printf("\tFound NodeToAnalyse: %s\n", NodeToAnalyse->InstructionString);
#endif

  // Create a Queue object and add NodeToAnalyse to front of Queue
  NodeProcessingQItem *QItem;
  if( (QItem = (NodeProcessingQItem *)malloc((size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
    printf("#AF: Out of memory error!");
    exit(EXIT_FAILURE);
  }
  QItem->Node = NodeToAnalyse;
  QItem->NextItem = NULL;
  struct NodeProcessingQ ProcessingQ = {QItem, QItem, 1};
  NodeProcessingQItem *WRTNode = QItem;

  // Store Amplification factor of NodeToAnalyse with Itself in table as 1
  AFResult->DoubleAFItemPointer->ResultInstructionString = NodeToAnalyse->InstructionString;
  AFResult->DoubleAFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
  AFResult->DoubleAFItemPointer->WRTInstructionString = WRTNode->Node->InstructionString;
  AFResult->DoubleAFItemPointer->WRTNodeID = WRTNode->Node->NodeId;
  AFResult->DoubleAFItemPointer->AF = 1;
  AFResult->DoubleAFRecords++;

  // Amplification Factor Calculation
  // Loop till Queue is empty -> Front == Back
  while(ProcessingQ.Size != 0 && ProcessingQ.Front != NULL && ProcessingQ.Back != NULL) {
#if FAF_DEBUG==2
    NodeProcessingQItem *ProcessingQItemPointer;
    printf("\n\tPrinting the Node Processing Q of size: %lu\n", ProcessingQ.Size);
    printf("\tQ Front - Node Id: %d, Instruction String: %s\n",
           ProcessingQ.Front->Node->NodeId,
           ProcessingQ.Front->Node->InstructionString);
    printf("\tQ Back - Node Id: %d, Instruction String: %s\n",
           ProcessingQ.Back->Node->NodeId,
           ProcessingQ.Back->Node->InstructionString);
    for (ProcessingQItemPointer = ProcessingQ.Front;
         ProcessingQItemPointer!=NULL;
         ProcessingQItemPointer = ProcessingQItemPointer->NextItem) {
      printf("\t\tNode Id: %d, Instruction String: %s\n",
             ProcessingQItemPointer->Node->NodeId,
             ProcessingQItemPointer->Node->InstructionString);
    }
#endif
    // Get the front of Q
    WRTNode = ProcessingQ.Front;

    // From AF Table, get AF at NodeToAnalyse for WRTNode
    DoubleAFItem *AFItemPointer;
    uint64_t RecordCounter;

#if FAF_DEBUG==2
    printf("\n\tFinding AF at NodeToAnalyse %d WRTNode %d\n",
           NodeToAnalyse->NodeId, WRTNode->Node->NodeId);
#endif
    for(RecordCounter = 0, AFItemPointer = AFResult->DoubleAFItemPointer;
         RecordCounter < AFResult->DoubleAFRecords && AFItemPointer->WRTNodeID != WRTNode->Node->NodeId;
         RecordCounter++, AFItemPointer++) {
#if FAF_DEBUG==2
      printf("\t\tAF@Node Id: %d, WRT Node Id: %d is AF=%f\n",
             AFItemPointer->ResultNodeID, AFItemPointer->WRTNodeID, AFItemPointer->AF);
#endif
    }

#if FAF_DEBUG
    printf("\tRequired AF@Node Id: %d, WRT Node Id: %d is AF=%f\n",
           AFItemPointer->ResultNodeID, AFItemPointer->WRTNodeID, AFItemPointer->AF);
#endif

    double AFofWRTNode = AFItemPointer->AF;

    // Calculate AF and Push nodes to process on ProcessingQ
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
        AFItemPointer = &AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords];

        AFItemPointer->ResultInstructionString =
            NodeToAnalyse->InstructionString;
        AFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
        AFItemPointer->WRTInstructionString =
            WRTNode->Node->LeftNode->InstructionString;
        AFItemPointer->WRTNodeID = WRTNode->Node->LeftNode->NodeId;
        AFItemPointer->AF =
            AFofWRTNode *
            StorageTable->FP32ACItems[WRTNode->Node->NodeId].ACWRTX;

        AFResult->DoubleAFRecords++;

#if FAF_DEBUG == 2
        printf("\n\tNew AFItem Stored:\n");
        printf("\t\tInstruction String: %s\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .ResultInstructionString);
        printf("\t\tNode Id: %d\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .ResultNodeID);
        printf("\t\tWRTInstructionString: %s\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .WRTInstructionString);
        printf("\t\tWRTNodeID: %d\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .WRTNodeID);
        printf("\t\tAF: %f\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1].AF);
#endif
        if (WRTNode->Node->LeftNode->Kind != 0) {
          // Pushing Node on Processing Q
          if ((QItem = (NodeProcessingQItem *)malloc(
                   (size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
            printf("#AF: Out of memory error!");
            exit(EXIT_FAILURE);
          }
          QItem->Node = WRTNode->Node->LeftNode;
          QItem->NextItem = NULL;
          ProcessingQ.Back->NextItem = QItem;
          ProcessingQ.Back = ProcessingQ.Back->NextItem;

          ProcessingQ.Size++;
        }
      }
      break;
    case 2:
#if FAF_DEBUG
      printf("\tIn Binary case\n");
#endif

      // Multiply above AF with left AC and store in table
      if(WRTNode->Node->LeftNode != NULL) {
        AFItemPointer = &AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords];
        AFItemPointer->ResultInstructionString =
            NodeToAnalyse->InstructionString;
        AFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
        AFItemPointer->WRTInstructionString =
            WRTNode->Node->LeftNode->InstructionString;
        AFItemPointer->WRTNodeID = WRTNode->Node->LeftNode->NodeId;
        AFItemPointer->AF =
            AFofWRTNode *
            StorageTable->FP32ACItems[WRTNode->Node->NodeId].ACWRTX;
        AFResult->DoubleAFRecords++;

#if FAF_DEBUG == 2
        printf("\n\tNew AFItem Stored:\n");
        printf("\t\tInstruction String: %s\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .ResultInstructionString);
        printf("\t\tNode Id: %d\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .ResultNodeID);
        printf("\t\tWRTInstructionString: %s\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .WRTInstructionString);
        printf("\t\tWRTNodeID: %d\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .WRTNodeID);
        printf("\t\tAF: %f\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1].AF);
#endif

        if (WRTNode->Node->LeftNode->Kind != 0) {
          // Pushing Left Node on Processing Q
          if ((QItem = (NodeProcessingQItem *)malloc(
                   (size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
            printf("#AF: Out of memory error!");
            exit(EXIT_FAILURE);
          }
          QItem->Node = WRTNode->Node->LeftNode;
          QItem->NextItem = NULL;
          ProcessingQ.Back->NextItem = QItem;
          ProcessingQ.Back = ProcessingQ.Back->NextItem;

          ProcessingQ.Size++;
        }
      }

      // Multiply above AF with right AC and store in table
      if(WRTNode->Node->RightNode != NULL) {
        AFItemPointer = &AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords];

        AFItemPointer->ResultInstructionString =
            NodeToAnalyse->InstructionString;
        AFItemPointer->ResultNodeID = NodeToAnalyse->NodeId;
        AFItemPointer->WRTInstructionString =
            WRTNode->Node->RightNode->InstructionString;
        AFItemPointer->WRTNodeID = WRTNode->Node->RightNode->NodeId;
        AFItemPointer->AF =
            AFofWRTNode *
            StorageTable->FP32ACItems[WRTNode->Node->NodeId].ACWRTY;

        AFResult->DoubleAFRecords++;

#if FAF_DEBUG == 2
        printf("\n\tNew AFItem Stored:\n");
        printf("\t\tInstruction String: %s\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .ResultInstructionString);
        printf("\t\tNode Id: %d\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .ResultNodeID);
        printf("\t\tWRTInstructionString: %s\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .WRTInstructionString);
        printf("\t\tWRTNodeID: %d\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1]
                   .WRTNodeID);
        printf("\t\tAF: %f\n",
               AFResult->DoubleAFItemPointer[AFResult->DoubleAFRecords - 1].AF);
#endif

        if (WRTNode->Node->RightNode->Kind != 0) {
          // Pushing Right Node on Processing Q
          if ((QItem = (NodeProcessingQItem *)malloc(
                   (size_t)((int64_t)sizeof(NodeProcessingQItem)))) == NULL) {
            printf("#AF: Out of memory error!");
            exit(EXIT_FAILURE);
          }
          QItem->Node = WRTNode->Node->RightNode;
          QItem->NextItem = NULL;
          ProcessingQ.Back->NextItem = QItem;
          ProcessingQ.Back = ProcessingQ.Back->NextItem;

          ProcessingQ.Size++;
        }
      }

      // Popping off Front of Q
      ProcessingQ.Front = ProcessingQ.Front->NextItem;
      ProcessingQ.Size--;
      break;
    }
    free(WRTNode);
  }

#if FAF_DEBUG
  printf("\nDone Analysing fp64 NodeId: %d, Instruction: %s\n",
         NodeToAnalyse->NodeId, NodeToAnalyse->InstructionString);
#endif

  return ;
}

void fAFStoreResult() {
  // Create a directory if not present
  char *DirectoryName = (char *)malloc((strlen(LOG_DIRECTORY_NAME)+1) * sizeof(char));
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  char ExecutionId[5000];
  char FileName[5000];
  FileName[0] = '\0';
  strcpy(FileName, strcat(strcpy(FileName, DirectoryName), "/fAF_"));

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

  fprintf(FP, "\t\"FP32\": [");
  int I = 0;
  while ((uint64_t)I < AFResult->FloatAFRecords) {
    if (fprintf(FP,
                "\t\t{\n"
                "\t\t\t\"Result Instruction\": \"%s\",\n"
                "\t\t\t\"Result Node ID\":%d,\n"
                "\t\t\t\"AF WRT Node\": \"%s\",\n"
                "\t\t\t\"WRT Node ID\":%d,\n"
                "\t\t\t\"AF\": %f\n",
                AFResult->FloatAFItemPointer[I].ResultInstructionString,
                AFResult->FloatAFItemPointer[I].ResultNodeID,
                AFResult->FloatAFItemPointer[I].WRTInstructionString,
                AFResult->FloatAFItemPointer[I].WRTNodeID,
                AFResult->FloatAFItemPointer[I].AF) > 0)
      RecordsStored++;

    if (RecordsStored != AFResult->FloatAFRecords)
      fprintf(FP, "\t\t},\n");
    else
      fprintf(FP, "\t\t}\n");
    I++;
  }
  fprintf(FP, "\t],\n");

  RecordsStored = 0;

  fprintf(FP, "\t\"FP64\": [");
  I = 0;
  while ((uint64_t)I < AFResult->DoubleAFRecords) {
    if (fprintf(FP,
                "\t\t{\n"
                "\t\t\t\"Result Instruction\": \"%s\",\n"
                "\t\t\t\"Result Node ID\":%d,\n"
                "\t\t\t\"AF WRT Node\": \"%s\",\n"
                "\t\t\t\"WRT Node ID\":%d,\n"
                "\t\t\t\"AF\": %lf\n",
                AFResult->DoubleAFItemPointer[I].ResultInstructionString,
                AFResult->DoubleAFItemPointer[I].ResultNodeID,
                AFResult->DoubleAFItemPointer[I].WRTInstructionString,
                AFResult->DoubleAFItemPointer[I].WRTNodeID,
                AFResult->DoubleAFItemPointer[I].AF) > 0)
      RecordsStored++;

    if (RecordsStored != AFResult->DoubleAFRecords)
      fprintf(FP, "\t\t},\n");
    else
      fprintf(FP, "\t\t}\n");
    I++;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");

  fclose(FP);
}

#endif // LLVM_AMPLIFICATIONFACTOR_H
