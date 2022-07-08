//
// Created by tanmay on 6/21/22.
//

#ifndef LLVM_COMPUTATIONGRAPH_H
#define LLVM_COMPUTATIONGRAPH_H

#include "AtomicCondition.h"

enum NodeKind {
  Register,
  UnaryInstruction,
  BinaryInstruction,
  Phi
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

typedef struct PhiNode {
  char *ResultRegister;
  char *ResidentInBB;
  int NumBranches;
  char **IncomingVals;
  int *IsConstant;
  char **BasicBlocks;
  struct PhiNode *Next;
} PhiNode;

typedef struct ComputationGraph {
  uint64_t LinkedListSize;
  CGNode *NodesLinkedListHead;
  InstructionNodePair* InstructionNodeMapHead;
  char **BasicBlockExecutionChainTail;
  uint64_t BasicBlockExecutionChainSize;
  PhiNode *PhiNodesListHead;
  uint64_t PhiNodesListSize;
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
  CGObject->LinkedListSize=0;

  // Allocate memory to the Instruction Node Map
  if( (CGObject->InstructionNodeMapHead =
           (struct InstructionNodePair *)malloc((size_t)((int64_t)sizeof(InstructionNodePair) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocate memory to the Basic Block Execution Chain
  if( (CGObject->BasicBlockExecutionChainTail =
           (char **)malloc((size_t)((int64_t)sizeof(char*) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }
  CGObject->BasicBlockExecutionChainTail--;
  CGObject->BasicBlockExecutionChainSize=0;

  // Allocate memory to the Phi Nodes List
  if( (CGObject->PhiNodesListHead =
           (struct PhiNode *)malloc((size_t)((int64_t)sizeof(PhiNode) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }
  CGObject->PhiNodesListSize=0;

  CG = CGObject;
}

int fCGnodesEqual(CGNode *Node1, CGNode *Node2)
{
  if (Node1->NodeId == Node2->NodeId)
    return 1;
  return 0;
}

int fCGNamedRegister(char *Register) {
  if (isdigit(Register[0]))
    return 1;
  return 0;
}

int fCGisConstant(char* Value) {
  if(Value[0] != '%')
    return 1;
  return 0;
}

int fCGcheckPHIInstruction(char *Register) {
  if(CG->PhiNodesListSize == 0)
    return 0;

  PhiNode *CurrNode = CG->PhiNodesListHead;
  while(CurrNode != NULL && strcmp(CurrNode->ResultRegister, Register)!=0) {
    CurrNode = CurrNode->Next;
  }
  if(CurrNode != NULL)
    return 1;
  return 0;
}

void fCGrecordPHIInstruction(char *InstructionString, char *ResidentBBName) {
  char *ResultReg;
  int *IsConstant;
  char **IncomingVals, **BasicBlocks;
  char *CharFindingPointer = InstructionString;
  int NumIncomingBranches;
  for (NumIncomingBranches=0; CharFindingPointer[NumIncomingBranches];
       CharFindingPointer[NumIncomingBranches]=='[' ? NumIncomingBranches++ : *CharFindingPointer++);
  const unsigned long ResultRegLen = strstr(InstructionString, " ") - InstructionString;

  // Allocating Memory
  ResultReg = (char*)malloc ( (ResultRegLen) * sizeof (char));
  IsConstant = (int*)malloc ( (NumIncomingBranches) * sizeof (int));
  IncomingVals = (char**)malloc ( (NumIncomingBranches) * sizeof (char));
  BasicBlocks = (char**)malloc ( (NumIncomingBranches) * sizeof (char));

  // Copying the Register Name, ignoring the '%'
  strncpy(ResultReg, InstructionString+1, ResultRegLen);
  ResultReg[ResultRegLen]=0;

  printf("Instruction String:%s\n", InstructionString);
  printf("ResultReg:%s\n", ResultReg);

  CharFindingPointer = InstructionString;
  // Collecting the Registers and Basic Blocks
  for(int CurrentBranchIndex=0;
       CurrentBranchIndex<NumIncomingBranches; CurrentBranchIndex++) {
    char *IncomingValueString;
    char *BasicBlockString;

    // Copying Incoming Value
    CharFindingPointer = strstr(CharFindingPointer, "[") + 2;
    if (fCGisConstant(CharFindingPointer))
      *(IsConstant+CurrentBranchIndex)=1;
    else
      *(IsConstant+CurrentBranchIndex)=0;
    const unsigned long IncomingValueLen = (strstr(CharFindingPointer, " ") - CharFindingPointer);
    IncomingValueString = (char*)malloc ( IncomingValueLen * sizeof (char));
    strncpy(IncomingValueString, CharFindingPointer, IncomingValueLen-1);
    IncomingValueString[IncomingValueLen-1]=0;

    // Copying Basic Block Name
    CharFindingPointer = strstr(CharFindingPointer, ",") + 3;
    const unsigned long BasicBlockLen = (strstr(CharFindingPointer, " ") - CharFindingPointer+1);
    BasicBlockString = (char*)malloc ( BasicBlockLen * sizeof (char));
    strncpy(BasicBlockString, CharFindingPointer, BasicBlockLen-1);
    BasicBlockString[BasicBlockLen-1]=0;

    *(IncomingVals+CurrentBranchIndex)=IncomingValueString;
    *(BasicBlocks+CurrentBranchIndex)=BasicBlockString;

    printf("IncomingValue:%s\n", *(IncomingVals+CurrentBranchIndex));
    printf("BasicBlock:%s\n", *(BasicBlocks+CurrentBranchIndex));
  }
  printf("\n");

  if(fCGcheckPHIInstruction(ResultReg))
    return ;

  PhiNode *Node = NULL;
  PhiNode *CurrNode = NULL;
  PhiNode *PrevNode = NULL;

  if((Node = (PhiNode *)malloc(sizeof(PhiNode))) == NULL) {
    printf("#fAC: PhiNodeList out of memory!");
    exit(EXIT_FAILURE);
  }

  Node->ResultRegister = ResultReg;
  Node->ResidentInBB = ResidentBBName;
  Node->NumBranches = NumIncomingBranches;
  Node->IncomingVals = IncomingVals;
  Node->BasicBlocks = BasicBlocks;
  Node->Next = NULL;

  CurrNode = CG->PhiNodesListHead;
  if(CG->PhiNodesListSize == 0) {
    CG->PhiNodesListHead = Node;
  }
  else {
    while (CurrNode != NULL) {
      PrevNode = CurrNode;
      CurrNode = CurrNode->Next;
    }
    CurrNode = Node;
    PrevNode->Next = CurrNode;
  }
  CG->PhiNodesListSize++;

  for(int CurrentBranchIndex=0;
       CurrentBranchIndex<NumIncomingBranches; CurrentBranchIndex++) {
    printf("NodeIncomingValue:%s\n", *(Node->IncomingVals+CurrentBranchIndex));
    printf("NodeBasicBlock:%s\n", *(Node->BasicBlocks+CurrentBranchIndex));
  }
}

void fCGrecordCurrentBasicBlock(char *BasicBlock) {
  CG->BasicBlockExecutionChainTail++;
  *CG->BasicBlockExecutionChainTail = BasicBlock;
  CG->BasicBlockExecutionChainSize++;
}

int fCGisPHIInstruction(char *InstructionString) {
  if(strstr(InstructionString, "phi")!=NULL)
    return 1;
  return 0;
}

void fCGprintStringArray(char **ArrayToPrint) {

}

char *fCGperformPHIResolution(char *PHIInstruction) {
  char *ResolvedInstruction;

  printf("\n");

  // Copying Register Name
  const unsigned long RegisterNameLen = (strstr(PHIInstruction, " ") - PHIInstruction);
  char *ResolvedInstructionRegister = (char*)malloc ( RegisterNameLen * sizeof (char));
  strncpy(ResolvedInstructionRegister, PHIInstruction+1, RegisterNameLen);
  ResolvedInstructionRegister[RegisterNameLen]=0;
  printf("PHIRegister Name:%s\n", ResolvedInstructionRegister);

  char **PreviousBasicBlock = CG->BasicBlockExecutionChainTail-1;

  // Finding Instruction in Phi Node list
  PhiNode *CurrPhiNode = CG->PhiNodesListHead;
  printf("PHIRegister Name2:%s\n", ResolvedInstructionRegister);
  while(CurrPhiNode != NULL && strcmp(CurrPhiNode->ResultRegister, ResolvedInstructionRegister) != 0) {
    printf("PHIRegister Name:%s\n", CurrPhiNode->ResultRegister);
    CurrPhiNode = CurrPhiNode->Next;
  }
  printf("PhiNodeFound:%s\n", CurrPhiNode->ResultRegister);

  // While loop till instruction is no longer phi
  while (1) {
    // Resolve Phi
    int *IsConstant = CurrPhiNode->IsConstant;
    char **IncomingValsUnit = CurrPhiNode->IncomingVals;
    char **BasicBlocksUnit = CurrPhiNode->BasicBlocks;

    int BranchCounter;
    printf("PreviousBasicBlock:%s\n", *PreviousBasicBlock);
    for(BranchCounter = 0;
         BranchCounter<CurrPhiNode->NumBranches && strcmp(*BasicBlocksUnit, *PreviousBasicBlock) != 0;
         BranchCounter++, IsConstant+=1, IncomingValsUnit+=1, BasicBlocksUnit+=1);

    char **BBPointer = CG->BasicBlockExecutionChainTail;
    printf("\nBasic Block Execution Chain:\n");
    for (uint64_t I = 0; I < CG->BasicBlockExecutionChainSize; ++I, BBPointer-=1)
      printf("%s\n", *BBPointer);
    printf("\n");

    if(BranchCounter == CurrPhiNode->NumBranches) {
      printf("#fCG: Branch not found in phi node. Check!\n");
      exit(EXIT_FAILURE);
    } else {
      ResolvedInstructionRegister = *IncomingValsUnit;
    }
    printf("Resolved Instruction Register:%s\n", ResolvedInstructionRegister);

    // Finding Instruction in Phi Node list
    CurrPhiNode = CG->PhiNodesListHead;
    while(CurrPhiNode != NULL && strncmp(CurrPhiNode->ResultRegister, ResolvedInstructionRegister+1, strlen(ResolvedInstructionRegister)-1) != 0) {
      printf("CurrPhiNode->ResultRegister:%s\n", CurrPhiNode->ResultRegister);
      CurrPhiNode = CurrPhiNode->Next;
    }

    // If phi node not found, exit while loop
    if(CurrPhiNode == NULL) {
      break;
    }
    printf("PhiNodeFound:%s\n", CurrPhiNode->ResultRegister);

    printf("CurrPhiNode->ResidentInBB:%s\n", CurrPhiNode->ResidentInBB);
    while(strcmp(CurrPhiNode->ResidentInBB, *PreviousBasicBlock) != 0)
      PreviousBasicBlock--;
    PreviousBasicBlock--;
  }
  printf("Looking for Instruction Reg %s in Node List of len %lu \n", ResolvedInstructionRegister, strlen(ResolvedInstructionRegister));


  // Find instruction in NodeList
  CGNode *CurrNode = CG->NodesLinkedListHead;
  while (CurrNode != NULL && strncmp(ResolvedInstructionRegister,
                                     CurrNode->InstructionString,
                                     strlen(ResolvedInstructionRegister))!=0) {
    printf("Instruction:%s\n", CurrNode->InstructionString);
    CurrNode = CurrNode->Next;
  }
  if (CurrNode != NULL)
    ResolvedInstruction = CurrNode->InstructionString;
  else
    ResolvedInstruction = ResolvedInstructionRegister;

  printf("Final Resolved Instruction:%s\n", ResolvedInstruction);

  return ResolvedInstruction;
}

void fCGcreateNode(char *InstructionString, char *LeftOpInstructionString, char *RightOpInstructionString, enum NodeKind NK){
  CGNode *Node=NULL;
  CGNode *CurrNode=NULL;
  CGNode *PrevNode=NULL;
  InstructionNodePair *NewPair=NULL;
  InstructionNodePair *CurrPair = NULL;
  InstructionNodePair *PrevPair = NULL;

  // Allocating memory for new CGNode and new InstructionNodePair
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

  int LeftOpNodeId=-1;
  int RightOpNodeId=-1;
  char *ResolvedLeftInstruction=LeftOpInstructionString;
  char *ResolvedRightInstruction=RightOpInstructionString;
  // Linking Left and Right operand nodes to Node if any
  switch (NK) {
  case 0:
    break;
  case 1:
    // Setting the Left Node
    if (strstr(LeftOpInstructionString, "phi")!=NULL) {
      // Resolve Phi
      ResolvedLeftInstruction = fCGperformPHIResolution(LeftOpInstructionString);

//      if (ResolvedLeftInstruction[0] == '%') {
//        CurrPair = CG->InstructionNodeMapHead;
//        while (CurrPair != NULL && strstr(CurrPair->InstructionString,
//                                          strrchr(ResolvedLeftInstruction, '[') + 2) !=
//                                       CurrPair->InstructionString) {
//          CurrPair = CurrPair->Next;
//        }
//        ResolvedLeftInstruction = CurrPair->InstructionString;
//      }
    }

    // If Resolved Instruction is not empty or NULL or a constant
    if(ResolvedLeftInstruction != NULL && strcmp(ResolvedLeftInstruction, "") != 0 && ResolvedLeftInstruction[0] == '%') {
      CurrPair = CG->InstructionNodeMapHead;
      while (CurrPair != NULL && strncmp(CurrPair->InstructionString,
                                        ResolvedLeftInstruction,
                                        strlen(ResolvedLeftInstruction)) != 0) {
        CurrPair = CurrPair->Next;
      }
      LeftOpNodeId = CurrPair->NodeId;

      CurrNode = CG->NodesLinkedListHead;
      while (CurrNode != NULL && CurrNode->NodeId != LeftOpNodeId) {
        CurrNode = CurrNode->Next;
      }
      Node->LeftNode = CurrNode;
    }

    if(Node->LeftNode != NULL) {
      Node->Height = Node->LeftNode->Height;
      Node->LeftNode->RootNode = 0;
    }
    Node->Height = Node->Height+1;
    Node->RootNode = 1;

    break;
  case 2:
    // Setting the Left Node
    printf("InstructionString: %s\n", InstructionString);
    printf("LeftOpInstructionString: %s\n", LeftOpInstructionString);
    printf("RightOpInstructionString: %s\n", RightOpInstructionString);
    printf("Current BasicBlockName: %s\n", *CG->BasicBlockExecutionChainTail);
    if (strcmp(*CG->BasicBlockExecutionChainTail, "entry") != 0)
      printf("Previous BasicBlockName: %s\n", *(CG->BasicBlockExecutionChainTail-1));

    // If LeftOperand's instruction is a phi node, resolve to a non-phi node
    if (strstr(LeftOpInstructionString, "phi")!=NULL) {
      // Resolve Phi
      ResolvedLeftInstruction = fCGperformPHIResolution(LeftOpInstructionString);

//      if (ResolvedLeftInstruction[0] == '%') {
//        CurrPair = CG->InstructionNodeMapHead;
//        while (CurrPair != NULL && strstr(CurrPair->InstructionString,
//                                          strrchr(ResolvedLeftInstruction, '[') + 2) !=
//                                       CurrPair->InstructionString) {
//          CurrPair = CurrPair->Next;
//        }
//        ResolvedLeftInstruction = CurrPair->InstructionString;
//      }
    }

    printf("ResolvedLeftInstruction: %s\n", ResolvedLeftInstruction);

    // If Resolved Instruction is not empty or NULL or a constant
    if(ResolvedLeftInstruction != NULL && strcmp(ResolvedLeftInstruction, "") != 0 && ResolvedLeftInstruction[0] == '%') {
      CurrPair = CG->InstructionNodeMapHead;
      while (CurrPair != NULL &&
             strncmp(CurrPair->InstructionString,
                     ResolvedLeftInstruction,
                     strlen(ResolvedLeftInstruction)) != 0) {
        printf("CurrPair->InstructionString: %s\n", CurrPair->InstructionString);
        CurrPair = CurrPair->Next;
      }
      LeftOpNodeId = CurrPair->NodeId;

      printf("LeftOpNodeId: %d\n", LeftOpNodeId);

      CurrNode = CG->NodesLinkedListHead;
      while (CurrNode != NULL && CurrNode->NodeId != LeftOpNodeId) {
        CurrNode = CurrNode->Next;
      }
      Node->LeftNode = CurrNode;
    }

    // Setting the Right Node
    if (strstr(RightOpInstructionString, "phi")!=NULL) {
      // Resolve Phi
      ResolvedRightInstruction = fCGperformPHIResolution(RightOpInstructionString);

//      if (ResolvedRightInstruction[0] == '%') {
//        CurrPair = CG->InstructionNodeMapHead;
//        while (CurrPair != NULL && strstr(CurrPair->InstructionString,
//                                          strrchr(ResolvedRightInstruction, '[') + 2) !=
//                                       CurrPair->InstructionString) {
//          CurrPair = CurrPair->Next;
//        }
//        ResolvedRightInstruction = CurrPair->InstructionString;
//      }
    }

    printf("ResolvedRightInstruction: %s\n", ResolvedRightInstruction);

    // If Resolved Instruction is not empty or NULL or a constant
    if(ResolvedRightInstruction != NULL && strcmp(ResolvedRightInstruction, "") != 0 && ResolvedRightInstruction[0] == '%') {
      CurrPair = CG->InstructionNodeMapHead;
      while (CurrPair != NULL && strncmp(CurrPair->InstructionString,
                                        ResolvedRightInstruction,
                                         strlen(ResolvedRightInstruction)) != 0) {
        CurrPair = CurrPair->Next;
      }
      RightOpNodeId = CurrPair->NodeId;

      printf("RightOpNodeId: %d\n", RightOpNodeId);

      CurrNode = CG->NodesLinkedListHead;
      while (CurrNode != NULL && CurrNode->NodeId != RightOpNodeId) {
        CurrNode = CurrNode->Next;
      }
      Node->RightNode = CurrNode;
    }

    if(Node->LeftNode != NULL) {
      Node->Height = Node->LeftNode->Height;
      Node->LeftNode->RootNode = 0;
    }
    if(Node->RightNode != NULL && Node->Height < Node->RightNode->Height) {
      Node->Height = Node->RightNode->Height;
      Node->RightNode->RootNode = 0;
    }
    Node->Height = Node->Height+1;
    Node->RootNode = 1;

    break;
  default:
    printf("#fAC: Node Kind Unknown!");
    exit(EXIT_FAILURE);
  }

  // Update/Insert a New Key-Value pair in InstructionNodeMap
  CurrPair = CG->InstructionNodeMapHead;
  PrevPair = NULL;
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

  printf("\n");
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

  // Building Graph
  FILE *FP = fopen(FileName, "w");
  fprintf(FP, "digraph ");
  fprintf(FP, "G ");
  fprintf(FP, "{\n");

  CGNode *CurrentNode = CG->NodesLinkedListHead;
  while (CurrentNode!=NULL) {
    switch (CurrentNode->Kind) {
    case 0:
      fprintf(FP, "\t%d [shape=rectangle];\n", CurrentNode->NodeId);
      break;
    case 1:
      if (CurrentNode->LeftNode != NULL)
        fprintf(FP, "\t%d -> %d;\n", CurrentNode->LeftNode->NodeId, CurrentNode->NodeId);
      break;
    case 2:
      if (CurrentNode->LeftNode != NULL)
        fprintf(FP, "\t%d -> %d;\n", CurrentNode->LeftNode->NodeId, CurrentNode->NodeId);
      if (CurrentNode->RightNode != NULL)
        fprintf(FP, "\t%d -> %d;\n", CurrentNode->RightNode->NodeId, CurrentNode->NodeId);
      break;
    default:
      break;
    }
    CurrentNode = CurrentNode->Next;
  }

  // Creating Legend
  fprintf(FP, "\tsubgraph cluster {\n");
  fprintf(FP, "\t\tnode [shape=plaintext];\n");
  fprintf(FP, "\t\tlabel = \"Legend\";\n");
  fprintf(FP, "\t\tkey [label=<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\">\n");
  CurrentNode = CG->NodesLinkedListHead;
  while (CurrentNode!=NULL) {
    fprintf(FP, "\t\t\t<tr><td>%d</td><td align=\"left\">%s</td></tr>\n",
            CurrentNode->NodeId, CurrentNode->InstructionString);
    CurrentNode = CurrentNode->Next;
  }
  fprintf(FP, "\t\t\t</table>>]\n");

  // Ending Legend
  fprintf(FP, "\t}\n");

  // Ending Digraph
  fprintf(FP, "}\n");

  fclose(FP);
}

void fAFAnalysis() {

  return ;
}
//void fAFAnalysis(char *InstructionString, char *InstructionToAnalyse, char *BasicBlockName, enum NodeKind NK) {
//
//  return ;
//}



#endif // LLVM_COMPUTATIONGRAPH_H
