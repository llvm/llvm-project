//
// Created by tanmay on 6/21/22.
//

#ifndef LLVM_COMPUTATIONGRAPH_H
#define LLVM_COMPUTATIONGRAPH_H

#include "AtomicCondition.h"

enum NodeKind {
  Number,
  Register,
  UnaryInstruction,
  BinaryInstruction
};

typedef struct CGNode {
  int NodeId;
  char *InstructionString;
  enum NodeKind Kind;
  struct CGNode *LeftNode;
  struct CGNode *RightNode;
  int Height;
  int RootNode;
  struct CGNode *Next;
} CGNode;

typedef struct InstructionNodePair {
  char *InstructionString;
  int NodeId;
  struct InstructionNodePair *Next;
} InstructionNodePair;

typedef struct ComputationGraph {
  uint64_t LinkedListSize;
  CGNode* NodesLinkedListHead;
  InstructionNodePair* InstructionNodeMapHead;
} ComputationGraph;

ComputationGraph *CG;

void fCGInitialize() {
  ComputationGraph *CGObject = NULL;
  int64_t Size = 1000;

  // Allocating the graph itself
  if(( CGObject = (ComputationGraph*)malloc(sizeof(ComputationGraph))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocate memory to the Nodes linked list
  if( (CGObject->NodesLinkedListHead =
           (struct CGNode *)malloc((size_t)((int64_t)sizeof(CGNode) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocate memory to the Instruction Node Map
  if( (CGObject->InstructionNodeMapHead =
           (struct InstructionNodePair *)malloc((size_t)((int64_t)sizeof(InstructionNodePair) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  CGObject->LinkedListSize=0;

  CG = CGObject;
}

int fCGnodesEqual(CGNode *Node1, CGNode *Node2)
{
  if (Node1->NodeId == Node2->NodeId)
    return 1;
  return 0;
}

void fCGcreateNode(char *InstructionString, char *LeftOpInstructionString, char *RightOpInstructionString, enum NodeKind NK){
  CGNode *Node=NULL;
  CGNode *CurrNode=NULL;
  CGNode *PrevNode=NULL;
  InstructionNodePair *NewPair=NULL;

  if((Node = (CGNode *)malloc(sizeof(CGNode))) == NULL) {
    printf("#fAC: AC table out of memory error!");
    exit(EXIT_FAILURE);
  }
  Node->NodeId = NodeCounter;
  Node->InstructionString = InstructionString;
  Node->Kind = NK;
  Node->LeftNode = NULL;
  Node->RightNode = NULL;
  Node->Height = 0;
  Node->RootNode = 1;
  Node->Next = NULL;

  if((NewPair = (InstructionNodePair *)malloc(sizeof(InstructionNodePair))) == NULL) {
    printf("#fAC: AC table out of memory error!");
    exit(EXIT_FAILURE);
  }
  NewPair->InstructionString = InstructionString;
  NewPair->NodeId = NodeCounter;
  NewPair->Next = NULL;

  // Update/Insert a New Key-Value pair in InstructionNodeMap
  InstructionNodePair *CurrPair = CG->InstructionNodeMapHead;
  InstructionNodePair *PrevPair = NULL;
  if (CG->LinkedListSize==0)
    CG->InstructionNodeMapHead = NewPair;
  else {
    while (CurrPair != NULL &&
           CurrPair->InstructionString != InstructionString) {
      PrevPair = CurrPair;
      CurrPair = CurrPair->Next;
    }

    if (CurrPair != NULL) {
      CurrPair->NodeId = NodeCounter;
    } else { // Nope, could't find it
      PrevPair->Next = NewPair;
    }
  }
  NodeCounter++;


  int LeftOpNodeId=-1;
  int RightOpNodeId=-1;
  // Linking Left and Right operand nodes to Node if any
  switch (NK) {
  case 0:
  case 1:
    break;
  case 2:
    CurrPair = CG->InstructionNodeMapHead;
    while(CurrPair != NULL && strcmp(CurrPair->InstructionString, LeftOpInstructionString)!=0) {
      CurrPair = CurrPair->Next;
    }
    LeftOpNodeId = CurrPair->NodeId;

    CurrNode = CG->NodesLinkedListHead;
    while(CurrNode != NULL && CurrNode->NodeId!=LeftOpNodeId) {
      CurrNode = CurrNode->Next;
    }

    Node->LeftNode = CurrNode;
    Node->Height = Node->LeftNode->Height+1;
    Node->LeftNode->RootNode = 0;
    Node->RootNode = 1;

    break;
  case 3:
    // Setting the Left Node
    CurrPair = CG->InstructionNodeMapHead;
    while(CurrPair != NULL && strcmp(CurrPair->InstructionString, LeftOpInstructionString) != 0) {
      CurrPair = CurrPair->Next;
    }
    LeftOpNodeId = CurrPair->NodeId;

    CurrNode = CG->NodesLinkedListHead;
    while(CurrNode != NULL && CurrNode->NodeId!=LeftOpNodeId) {
      CurrNode = CurrNode->Next;
    }
    Node->LeftNode = CurrNode;

    // Setting the Right Node
    CurrPair = CG->InstructionNodeMapHead;
    while(CurrPair != NULL && strcmp(CurrPair->InstructionString, RightOpInstructionString) != 0) {
      CurrPair = CurrPair->Next;
    }
    RightOpNodeId = CurrPair->NodeId;

    CurrNode = CG->NodesLinkedListHead;
    while(CurrNode != NULL && CurrNode->NodeId!=RightOpNodeId) {
      CurrNode = CurrNode->Next;
    }
    Node->RightNode = CurrNode;
    if(Node->LeftNode->Height > Node->RightNode->Height)
      Node->Height = Node->LeftNode->Height+1;
    else
      Node->Height = Node->RightNode->Height+1;
    Node->LeftNode->RootNode = 0;
    Node->RightNode->RootNode = 0;
    Node->RootNode = 1;

    break;
  default:
    printf("#fAC: Node Kind Unknown!");
    exit(EXIT_FAILURE);
  }
  // Adding Node to linked list
  if (CG->LinkedListSize==0)
    CG->NodesLinkedListHead = Node;
  else {
    CurrNode = CG->NodesLinkedListHead;
    while(CurrNode != NULL) {
      PrevNode = CurrNode;
      CurrNode = CurrNode->Next;
    }
    // We're at the end of the linked list
    PrevNode->Next = Node;
  }
  CG->LinkedListSize++;

  return ;
}

void fCGStoreResult() {
  // Create a directory
  struct stat ST;
  char DirectoryName[] = ".fAC_logs";
  if (stat(DirectoryName, &ST) == -1) {
    // TODO: Check the file mode and whether this one is the right one to use.
    mkdir(DirectoryName, 0775);
  }

  char ExecutionId[5000];
  char FileName[5000];
  FileName[0] = '\0';
  strcpy(FileName, ".fAC_logs/fCG_");

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

  fprintf(FP, "\t\"Nodes\": [");
  CGNode *CurrentNode = CG->NodesLinkedListHead;
  while (CurrentNode!=NULL) {
    fprintf(FP,
            "\t\t{\n"
            "\t\t\t\"NodeId\":%d,\n"
            "\t\t\t\"Instruction\":\"%s\",\n"
            "\t\t\t\"NodeKind\": %d,\n"
            "\t\t\t\"Height\": %d, \n"
            "\t\t\t\"RootNode\": %d, \n"
            "\t\t\t\"LeftNode\": %d,\n"
            "\t\t\t\"RightNode\": %d\n",
            CurrentNode->NodeId,
            CurrentNode->InstructionString,
            CurrentNode->Kind,
            CurrentNode->Height,
            CurrentNode->RootNode,
            (CurrentNode->LeftNode!=NULL?CurrentNode->LeftNode->NodeId:-1),
            (CurrentNode->RightNode!=NULL?CurrentNode->RightNode->NodeId:-1));

    if (CurrentNode->Next!=NULL)
      fprintf(FP, "\t\t},\n");
    else
      fprintf(FP, "\t\t}\n");
    CurrentNode = CurrentNode->Next;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");

  fclose(FP);

}

void fCGDotGraph() {
  // Create a directory
  struct stat ST;
  char DirectoryName[] = ".fAC_logs";
  if (stat(DirectoryName, &ST) == -1) {
    // TODO: Check the file mode and whether this one is the right one to use.
    mkdir(DirectoryName, 0775);
  }

  char ExecutionId[5000];
  char FileName[5000];
  FileName[0] = '\0';
  strcpy(FileName, ".fAC_logs/fCGDot_");

  fACGenerateExecutionID(ExecutionId);
  strcat(ExecutionId, ".gv");

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
  fprintf(FP, "digraph ");
  fprintf(FP, "G ");
  fprintf(FP, "{\n");

  CGNode *CurrentNode = CG->NodesLinkedListHead;
  while (CurrentNode!=NULL) {
    switch (CurrentNode->Kind) {
    case 0:
    case 1:
      fprintf(FP, "\t%d;\n", CurrentNode->NodeId);
      break;
    case 2:
      fprintf(FP, "\t%d -> %d;\n", CurrentNode->LeftNode->NodeId, CurrentNode->NodeId);
      break;
    case 3:
      fprintf(FP, "\t%d -> %d;\n", CurrentNode->LeftNode->NodeId, CurrentNode->NodeId);
      fprintf(FP, "\t%d -> %d;\n", CurrentNode->RightNode->NodeId, CurrentNode->NodeId);
      break;
    default:
      break;
    }
    CurrentNode = CurrentNode->Next;
  }

  fprintf(FP, "}\n");

  fclose(FP);
}

void fAFAnalysis() {


  return ;
}



#endif // LLVM_COMPUTATIONGRAPH_H
