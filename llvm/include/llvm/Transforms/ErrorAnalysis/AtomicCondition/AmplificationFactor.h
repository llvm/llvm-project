//
// Created by tanmay on 7/11/22.
//

#ifndef LLVM_AMPLIFICATIONFACTOR_H
#define LLVM_AMPLIFICATIONFACTOR_H

#include "ComputationGraph.h"

typedef struct NodeProcessingQItem {
  CGNode *Node;
  CGNode *NextItem;
} NodeProcessingQItem;

typedef struct NodeProcessingQ {
  NodeProcessingQItem *Front;
  NodeProcessingQItem *Back;
  uint64_t Size;
} NodeProcessingQ;

typedef struct FloatAFItem {
  char *ResultRegName;
  int ResultNodeID;
  char *WRTRegName;
  int WRTNodeID;
  float AF;
} FloatAFItem;

typedef struct DoubleAFItem {
  char *ResultRegName;
  int ResultNodeID;
  char *WRTRegName;
  int WRTNodeID;
  float AF;
} DoubleAFItem;

typedef struct AnalysisResult {
  NodeProcessingQ *ProcessingQ;
  FloatAFItem *FloatAFItemPointer;
  uint64_t FloatAFRecords;
  DoubleAFItem *DoubleAFItemPointer;
  uint64_t DoubleAFRecords;
} AnalysisResult;

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/
AnalysisResult *AFResult;

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
  AnalysisResult *AFObject = NULL;
  int64_t Size = 1000;

  // Allocating Memory to the analysis object itself
  if( (AFObject = (AnalysisResult *)malloc(sizeof(AnalysisResult))) == NULL) {
    printf("#AF: Queue out of memory error!");
    exit(EXIT_FAILURE);
  }
  AFObject->ProcessingQ = NULL;
  AFObject->FloatAFItemPointer = NULL;
  AFObject->DoubleAFItemPointer = NULL;
  AFObject->FloatAFRecords = 0;
  AFObject->DoubleAFRecords = 0;

  // Allocating Memory for Processing Queue
  if( (AFObject->ProcessingQ = (NodeProcessingQ *)malloc((size_t)((int64_t)sizeof(NodeProcessingQ) * Size))) == NULL) {
    printf("#AF: Queue out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocating Memory for Amplification Factor Tables
  if( (AFObject->FloatAFItemPointer = (FloatAFItem *)malloc((size_t)((int64_t)sizeof(FloatAFItem) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  if( (AFObject->DoubleAFItemPointer = (DoubleAFItem *)malloc((size_t)((int64_t)sizeof(DoubleAFItem) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  AFResult = AFObject;

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
  printf("\nPerforming Amplification Factor Calculation\n");
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
  CGNode *Node, *CurrCGNode;
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
       CurrCGNode != NULL && CurrCGNode->NodeId == NodeID;
       CurrCGNode=CurrCGNode->Next) {
#if FAF_DEBUG==2
    printf("\t\tInstruction String: %s, NodeID: %d\n", CurrCGNode->InstructionString, CurrCGNode->NodeId);
#endif
  }
  assert(CurrInstructionNode != NULL);
  Node = CurrCGNode;
#if FAF_DEBUG
  printf("\tFound Node: %s\n", Node->InstructionString);
#endif

  // Create a Queue object
  // Add Node to front of Queue
  // Store Amplification factor of Node with Itself in table as 1
  // ResNode = Node

  // Amplification Factor Calculation
  // Loop till Queue is empty -> Front == Back
    // Pop front of Queue and get CurrNode
    // From Table, get AF of ResNode with CurrNode
    // Multiply above AF with left AC and store in table
    // Multiply above AF with right AC and store in table
    // Push left and right in back of queue


  printf("Amplification Factor of NodeID is Value\n");
  return ;
}


void fAFfp64Analysis(char *ResultToAnalyse) {
#if FAF_DEBUG
  printf("Performing Amplification Factor Calculation\n");
  printf("\tResult to Analyse: %s\n", ResultToAnalyse);
#endif



  printf("Amplification Factor of NodeID is Value\n");
  return ;
}

void fAFStoreResult() {
//#if FAF_DEBUG

//#endif
}

#endif // LLVM_AMPLIFICATIONFACTOR_H
