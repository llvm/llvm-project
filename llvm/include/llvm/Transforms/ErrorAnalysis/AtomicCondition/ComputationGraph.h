//
// Created by tanmay on 6/21/22.
//

#ifndef LLVM_COMPUTATIONGRAPH_H
#define LLVM_COMPUTATIONGRAPH_H

#include "AtomicCondition.h"

enum NodeKind {
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

typedef struct PhiNode {
  char *PhiInstruction;
  char *ResidentInBB;
  int NumBranches;
  char **IncomingVals;
  char **BasicBlocks;
  struct PhiNode *Next;
} PhiNode;

typedef struct ComputationGraph {
  uint64_t LinkedListSize;
  CGNode *NodesLinkedListHead;
  InstructionNodePair* InstructionNodeMapHead;
  char **BasicBlockExecutionChainTail;
  PhiNode *PhiNodesListHead;
  uint64_t PhiNodesListSize;
} ComputationGraph;

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/

ComputationGraph *CG;

/*----------------------------------------------------------------------------*/
/* Utility Functions                                                          */
/*----------------------------------------------------------------------------*/

int fCGnodesEqual(CGNode *Node1, CGNode *Node2)
{
  if (Node1->NodeId == Node2->NodeId)
    return 1;
  return 0;
}

int fCGisRegister(char* Value) {
  if(Value[0] == '%')
    return 1;
  return 0;
}

int fCGNamedRegister(char *Register) {
  assert(fCGisRegister(Register));
  if(isdigit(Register[1]))
    return 1;
  return 0;
}

int fCGcheckPHIInstruction(char *Instruction) {
  if(CG->PhiNodesListSize == 0)
    return 0;

  PhiNode *CurrNode = CG->PhiNodesListHead;
  while(CurrNode != NULL && strcmp(CurrNode->PhiInstruction, Instruction)!=0) {
    CurrNode = CurrNode->Next;
  }
  if(CurrNode != NULL)
    return 1;
  return 0;
}

int fCGisPHIInstruction(char *InstructionString) {
  if(strstr(InstructionString, "phi")!=NULL)
    return 1;
  return 0;
}

void fCGprintStringArray(char **ArrayToPrint) {

}

void fCGInitialize() {
  ComputationGraph *CGObject = NULL;
  int64_t Size = 1000000;

  // Allocating the graph itself
  if(( CGObject = (ComputationGraph*)malloc(sizeof(ComputationGraph))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocate memory to the Nodes linked list
  if( (CGObject->NodesLinkedListHead =
           (CGNode *)malloc((size_t)((int64_t)sizeof(CGNode) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }
  CGObject->LinkedListSize=0;

  // Allocate memory to the Instruction Node Map
  if( (CGObject->InstructionNodeMapHead =
           (InstructionNodePair *)malloc((size_t)((int64_t)sizeof(InstructionNodePair) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }

  // Allocate memory to the Basic Block Execution Chain
  if( (CGObject->BasicBlockExecutionChainTail =
           (char **)malloc((size_t)((int64_t)sizeof(char*) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }
  // The first pointer will never be assigned and will remain NULL acting as a
  // boundary/terminator for anything traversing the chain in reverse.
  char **BasicBlockPointer = CGObject->BasicBlockExecutionChainTail;
  for (int Index = 0; Index < Size; ++Index, BasicBlockPointer++) {
    *BasicBlockPointer = NULL;
  }

  // Allocate memory to the Phi Nodes List
  if( (CGObject->PhiNodesListHead =
           (PhiNode *)malloc((size_t)((int64_t)sizeof(PhiNode) * Size))) == NULL) {
    printf("#CG: graph out of memory error!");
    exit(EXIT_FAILURE);
  }
  CGObject->PhiNodesListSize=0;

  CG = CGObject;

#if FAF_DEBUG
  // Create a directory if not present
  char *DirectoryName = (char *)malloc(strlen(LOG_DIRECTORY_NAME) * sizeof(char));
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);

  char ExecutionId[5000];
  char ACFileName[5000];
  char CGFileName[5000];
  char CGDotFileName[5000];
  ACFileName[0] = CGFileName[0] = CGDotFileName[0] = '\0';
  strcpy(ACFileName, strcat(strcpy(ACFileName, DirectoryName), "/program_log_"));
  strcpy(CGFileName, strcat(strcpy(CGFileName, DirectoryName), "/program_log_"));
  strcpy(CGDotFileName, strcat(strcpy(CGDotFileName, DirectoryName), "/program_log_"));

  fACGenerateExecutionID(ExecutionId);
  strcat(ExecutionId, ".txt");

  strcat(ACFileName, ExecutionId);
  strcat(CGFileName, ExecutionId);
  strcat(CGDotFileName, ExecutionId);

  printf("Atomic Conditions Storage File:%s\n", ACFileName);
  printf("Computation Graph Storage File:%s\n", CGFileName);
  printf("Dot Graph File:%s\n", CGDotFileName);
#endif
}

void fCGrecordPHIInstruction(char *InstructionString, char *ResidentBBName) {
  char *PhiInstruction;
  char **IncomingVals, **BasicBlocks;
  char *CharFindingPointer = InstructionString;
  int NumIncomingBranches;

  // Counting number of incoming branches in PHI instruction
  for (NumIncomingBranches=0; CharFindingPointer[NumIncomingBranches];
       CharFindingPointer[NumIncomingBranches]=='[' ? NumIncomingBranches++ : *CharFindingPointer++);

  // Allocating Memory
  IncomingVals = (char**)malloc ( (NumIncomingBranches) * sizeof (char));
  BasicBlocks = (char**)malloc ( (NumIncomingBranches) * sizeof (char));

  PhiInstruction=InstructionString;

#if FAF_DEBUG>=2
  printf("Recording PHI Instruction\n");
  printf("\tInstruction String:%s\n", InstructionString);
#endif

  CharFindingPointer = InstructionString;
  // Collecting the Registers and Basic Blocks by looping through all branches
  for(int CurrentBranchIndex=0;
       CurrentBranchIndex<NumIncomingBranches; CurrentBranchIndex++) {
    char *IncomingValueString;
    char *BasicBlockString;

    // Copying Incoming Value
    CharFindingPointer = strstr(CharFindingPointer, "[") + 2;
    const unsigned long IncomingValueLen = (strstr(CharFindingPointer, " ") - CharFindingPointer);
    IncomingValueString = (char*)malloc ( IncomingValueLen * sizeof (char));
    strncpy(IncomingValueString, CharFindingPointer, IncomingValueLen-1);
    IncomingValueString[IncomingValueLen-1]=0;

    // Copying Basic Block Name
    CharFindingPointer = strstr(CharFindingPointer, ",") + 2;
    const unsigned long BasicBlockLen = (strstr(CharFindingPointer, " ") - CharFindingPointer+1);
    BasicBlockString = (char*)malloc ( BasicBlockLen * sizeof (char));
    strncpy(BasicBlockString, CharFindingPointer, BasicBlockLen-1);
    BasicBlockString[BasicBlockLen-1]=0;

    *(IncomingVals+CurrentBranchIndex)=IncomingValueString;
    *(BasicBlocks+CurrentBranchIndex)=BasicBlockString;
  }

  if(fCGcheckPHIInstruction(PhiInstruction))
    return ;

  PhiNode *Node = NULL;
  PhiNode *CurrNode = NULL;
  PhiNode *PrevNode = NULL;

  if((Node = (PhiNode *)malloc(sizeof(PhiNode))) == NULL) {
    printf("#fAC: PhiNodeList out of memory!");
    exit(EXIT_FAILURE);
  }

  Node->PhiInstruction = PhiInstruction;
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

#if FAF_DEBUG>=2
  printf("\tRecorded for %s:\n", Node->PhiInstruction);
  for(int CurrentBranchIndex=0;
       CurrentBranchIndex < NumIncomingBranches; CurrentBranchIndex++) {
    printf("\t\tBranch %d:\n", CurrentBranchIndex);
    printf("\t\t\tIncoming Value: %s\n", *(Node->IncomingVals+CurrentBranchIndex));
    printf("\t\t\tBasic Block: %s\n", *(Node->BasicBlocks+CurrentBranchIndex));
  }
  printf("\n");
#endif
}

void fCGrecordCurrentBasicBlock(char *BasicBlock) {
  assert(fCGisRegister(BasicBlock));
  CG->BasicBlockExecutionChainTail++;
  *CG->BasicBlockExecutionChainTail = BasicBlock;

#if FAF_DEBUG>=2
  printf("Recording Basic Block: %s\n\n", BasicBlock);
#endif
}

// Backtracks the Basic Block Execution Chain to Resolve the PHI Instruction to
// a Non-PHI Instruction OR a Constant.
char *fCGperformPHIResolution(char *PHIInstruction) {
  // Ensuring we start with a PHI Instruction
  assert(fCGisPHIInstruction(PHIInstruction));

#if FAF_DEBUG
  printf("\nPerforming PHI Instruction Resolution to a Non-PHI Instruction\n");
#endif

  // Copying PHI Instruction Register Name
  const unsigned long PhiInstructionLen = strlen(PHIInstruction);
  char *ResolvedValue = (char*)malloc ( (PhiInstructionLen+1) * sizeof (char));
  strncpy(ResolvedValue, PHIInstruction, PhiInstructionLen);
  ResolvedValue[PhiInstructionLen]=0;

#if FAF_DEBUG
  printf("\tFirst PHI Instruction:%s\n", ResolvedValue);
#endif

  // Setting Basic Block Execution trace pointer to the tail of Basic Block
  // Execution chain
  char **PreviousBasicBlock = CG->BasicBlockExecutionChainTail-1;

  // Finding Instruction record in Phi Nodes List
#if FAF_DEBUG>=2
  printf("\tSearching for record of FIRST PHI Instruction in PHI Nodes List\n");
#endif

  PhiNode *CurrPhiNode = CG->PhiNodesListHead;
  while(CurrPhiNode != NULL && strcmp(CurrPhiNode->PhiInstruction, ResolvedValue) != 0) {
#if FAF_DEBUG>=2
    printf("\t\tPHI Nodes List: Current PHI Instruction Register Name:%s\n", CurrPhiNode->PhiInstruction);
#endif
    CurrPhiNode = CurrPhiNode->Next;
  }
  // The PHI Instruction is already recorded before it is invoked.
  assert(CurrPhiNode != NULL);
#if FAF_DEBUG>=2
  printf("\t\tRecord found! - %s\n", CurrPhiNode->PhiInstruction);
#endif

  // Loop to traverse the PHI Chain in reverse order of execution till a Non-PHI
  // Instruction is found
#if FAF_DEBUG>=2
  printf("\n\tBackTracking Non-PHI Instruction\n");
#endif

  while (1) {
    // Setting Pointers to traverse PHI Node's basic blocks and corresponding
    // incoming values
    char **IncomingValsUnit = CurrPhiNode->IncomingVals;
    char **BasicBlocksUnit = CurrPhiNode->BasicBlocks;

    // Finding the Basic Block in the PHI Instruction record
    int BranchCounter;
#if FAF_DEBUG>=2
    printf("\t\tPrevious Basic Block:%s\n", *PreviousBasicBlock);
#endif

    for(BranchCounter = 0;
         BranchCounter < CurrPhiNode->NumBranches && strcmp(*BasicBlocksUnit, *PreviousBasicBlock) != 0;
         BranchCounter++, IncomingValsUnit++, BasicBlocksUnit++);

    // Debugging section printing the Basic Block Execution Chain
#if FAF_DEBUG>=2
    char **BBPointer = CG->BasicBlockExecutionChainTail;
    printf("\n\t\tReversed Basic Block Execution Chain\n");
    while(*BBPointer != NULL) {
      printf("\t\t\tBasic Block Execution Chain: %s\n", *BBPointer);
      BBPointer--;
    }
    printf("\n");
#endif

    // In no case can the basic block NOT be found so the branch counter remains
    // less than the number of branches.
    assert(BranchCounter < CurrPhiNode->NumBranches);

    // Assuming the correct branch is chosen, saving the Incoming Register
    ResolvedValue = *IncomingValsUnit;

#if FAF_DEBUG>=2
    printf("\t\tResolved Value:%s\n", ResolvedValue);
#endif

    // Finding Instruction record in Phi Nodes List
#if FAF_DEBUG>=2
    printf("\t\tSearching for record of PHI Instruction in PHI Nodes List\n");
#endif
    CurrPhiNode = CG->PhiNodesListHead;
    while(CurrPhiNode != NULL && strncmp(CurrPhiNode->PhiInstruction, ResolvedValue, strlen(ResolvedValue)) != 0) {
#if FAF_DEBUG>=2
      printf("\t\t\tPHI Nodes List: Current PHI Instruction Register Name:%s\n", CurrPhiNode->PhiInstruction);
#endif
      CurrPhiNode = CurrPhiNode->Next;
    }

    // If phi node not found, exit while loop
    if(CurrPhiNode == NULL) {
#if FAF_DEBUG>=2
      printf("\t\tPHI Node not found corresponding to %s\n", ResolvedValue);
#endif
      break;
    }

#if FAF_DEBUG>=2
    printf("\t\t\tRecord found! - %s\n", CurrPhiNode->PhiInstruction);
#endif

    // Jumping one or more Basic Blocks in the Basic Block Execution Chain to
    // reach a Basic Block where the ResolvedValue resides.
#if FAF_DEBUG>=2
    printf("\t\t%s resides in Basic Block %s\n", ResolvedValue, CurrPhiNode->ResidentInBB);
#endif
    while(strcmp(CurrPhiNode->ResidentInBB, *PreviousBasicBlock) != 0)
      PreviousBasicBlock--;
    PreviousBasicBlock--;
  }

#if FAF_DEBUG
  printf("\n\tLooking for Register %s in Nodes List.\n", ResolvedValue);
#endif

  // Finding instruction in Nodes List
  CGNode *CurrNode = CG->NodesLinkedListHead;
  while (CurrNode != NULL && strncmp(ResolvedValue,
                                     CurrNode->InstructionString,
                                     strlen(ResolvedValue))!=0) {

#if FAF_DEBUG
    printf("\t\tNodes List:%s\n", CurrNode->InstructionString);
#endif
    CurrNode = CurrNode->Next;
  }

  if(CurrNode != NULL)
    ResolvedValue = CurrNode->InstructionString;
  else
    assert(!fCGisRegister(ResolvedValue));

#if FAF_DEBUG
  printf("\tFinal Resolved Value:%s\n\n", ResolvedValue);
#endif
  return ResolvedValue;
}

void fCGcreateNode(char *InstructionString, char *LeftOpInstructionString, char *RightOpInstructionString, enum NodeKind NK){
#if FAF_DEBUG
  printf("Creating New Node in Computation graph\n");
  printf("\tInstruction String: %s\n", InstructionString);
  printf("\tLeft Instruction String: %s\n", LeftOpInstructionString);
  printf("\tRight Instruction String: %s\n", RightOpInstructionString);
  printf("\tNode Kind: %d\n", NK);
  printf("\tCurrent BasicBlockName: %s\n", *CG->BasicBlockExecutionChainTail);
  if (strcmp(*CG->BasicBlockExecutionChainTail, "%entry") != 0)
    printf("\tPrevious BasicBlockName: %s\n", *(CG->BasicBlockExecutionChainTail-1));
#endif

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
  char *ResolvedLeftValue=LeftOpInstructionString;
  char *ResolvedRightValue=RightOpInstructionString;

  // Linking Left and Right operand nodes to Node if any
  switch (NK) {
  case 0:
    break;
  case 1:
    // Setting the Left Node
    if (fCGisPHIInstruction(LeftOpInstructionString)) {
#if FAF_DEBUG
      printf("\tResolving %s\n", LeftOpInstructionString);
#endif
      ResolvedLeftValue = fCGperformPHIResolution(LeftOpInstructionString);
    }

    // If Resolved Value is an Instruction
    if(fCGisRegister(ResolvedLeftValue)) {
#if FAF_DEBUG>=2
      printf("\n\tSearching for %s in Instruction->Node Map\n", ResolvedLeftValue);
#endif
      CurrPair = CG->InstructionNodeMapHead;
      while (CurrPair != NULL && strncmp(CurrPair->InstructionString,
                                        ResolvedLeftValue,
                                        strlen(ResolvedLeftValue)) != 0) {
#if FAF_DEBUG>=2
        printf("\t\tCurrent Pair: %s->%d\n", CurrPair->InstructionString, CurrPair->NodeId);
#endif
        CurrPair = CurrPair->Next;
      }
      assert(CurrPair != NULL);
      LeftOpNodeId = CurrPair->NodeId;
#if FAF_DEBUG>=2
      printf("\t\tPair Found: %s->%d\n", CurrPair->InstructionString, CurrPair->NodeId);
#endif

      CurrNode = CG->NodesLinkedListHead;
      while (CurrNode != NULL && CurrNode->NodeId != LeftOpNodeId) {
        CurrNode = CurrNode->Next;
      }
      assert(CurrNode != NULL);
      Node->LeftNode = CurrNode;
    }

    // If Left Node has been linked, update this nodes height correspondingly
    // and mark the Left Node as NOT a Root Node.
    if(Node->LeftNode != NULL) {
      Node->Height = Node->LeftNode->Height;
      Node->LeftNode->RootNode = 0;
    }
    Node->Height = Node->Height+1;
    Node->RootNode = 1;
    break;
  case 2:
    // Setting the Left Node
    // If LeftOperand's instruction is a phi node, resolve to a non-phi node
    if (fCGisPHIInstruction(LeftOpInstructionString)) {
#if FAF_DEBUG
      printf("\tResolving %s\n", LeftOpInstructionString);
#endif
      ResolvedLeftValue = fCGperformPHIResolution(LeftOpInstructionString);
    }

    // If Resolved Value is an Instruction
    if(fCGisRegister(ResolvedLeftValue)) {
#if FAF_DEBUG>=2
      printf("\n\tSearching for %s in Instruction->Node Map\n", ResolvedLeftValue);
#endif
      CurrPair = CG->InstructionNodeMapHead;
      while (CurrPair != NULL &&
             strncmp(CurrPair->InstructionString,
                     ResolvedLeftValue,
                     strlen(ResolvedLeftValue)) != 0) {
#if FAF_DEBUG>=2
        printf("\t\tCurrent Pair: %s->%d\n", CurrPair->InstructionString, CurrPair->NodeId);
#endif
        CurrPair = CurrPair->Next;
      }
      assert(CurrPair != NULL);
      LeftOpNodeId = CurrPair->NodeId;
#if FAF_DEBUG>=2
      printf("\t\tPair Found: %s->%d\n", CurrPair->InstructionString, CurrPair->NodeId);
#endif

      CurrNode = CG->NodesLinkedListHead;
      while (CurrNode != NULL && CurrNode->NodeId != LeftOpNodeId) {
//        printf("CurrNode Instruction: %s, Node ID: %d\n", CurrNode->InstructionString, CurrNode->NodeId);
        CurrNode = CurrNode->Next;
      }
      assert(CurrNode != NULL);
      Node->LeftNode = CurrNode;
    }

    // Setting the Right Node
    if (fCGisPHIInstruction(RightOpInstructionString)) {
#if FAF_DEBUG
      printf("\tResolving %s\n", RightOpInstructionString);
#endif
      ResolvedRightValue = fCGperformPHIResolution(RightOpInstructionString);
    }

    // If Resolved Value is an Instruction
    if(fCGisRegister(ResolvedRightValue)) {
#if FAF_DEBUG>=2
      printf("\n\tSearching for %s in Instruction->Node Map\n", ResolvedRightValue);
#endif
      CurrPair = CG->InstructionNodeMapHead;
      while (CurrPair != NULL && strncmp(CurrPair->InstructionString,
                                        ResolvedRightValue,
                                         strlen(ResolvedRightValue)) != 0) {
#if FAF_DEBUG>=2
        printf("\t\tCurrent Pair: %s->%d\n", CurrPair->InstructionString, CurrPair->NodeId);
#endif
        CurrPair = CurrPair->Next;
      }
      assert(CurrPair != NULL);
      RightOpNodeId = CurrPair->NodeId;
#if FAF_DEBUG>=2
      printf("\t\tPair Found: %s->%d\n", CurrPair->InstructionString, CurrPair->NodeId);
#endif

      CurrNode = CG->NodesLinkedListHead;
      while (CurrNode != NULL && CurrNode->NodeId != RightOpNodeId) {
        CurrNode = CurrNode->Next;
      }
      assert(CurrNode != NULL);
      Node->RightNode = CurrNode;
    }

    // If Left Node and/or Right Node have been linked, update this nodes height
    // correspondingly and mark the Left AND Right Nodes as NOT a Root Node.
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

#if FAF_DEBUG
  printf("\tNew Node Created\n");
  printf("\t\tNode ID: %d\n", Node->NodeId);
  printf("\t\tInstruction: %s\n", Node->InstructionString);
  printf("\t\tNode Type: %d\n", Node->Kind);
  if(Node->LeftNode!=NULL)
    printf("\t\tResolved Left Instruction: %s\n", Node->LeftNode->InstructionString);
  if(Node->RightNode!=NULL)
    printf("\t\tResolved Right Instruciton: %s\n", Node->RightNode->InstructionString);
  printf("\t\tNode Height: %d\n", Node->Height);
  printf("\t\tIs a Root Node? %s\n", Node->RootNode?"yes":"no");
  printf("\tNode Creation Completed\n\n");
#endif

  return ;
}

void fCGStoreResult() {
#if FAF_DEBUG
  // Create a directory if not present
  char *DirectoryName = (char *)malloc(strlen(LOG_DIRECTORY_NAME) * sizeof(char));
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  char ExecutionId[5000];
  char FileName[5000];
  FileName[0] = '\0';
  strcpy(FileName, strcat(strcpy(FileName, DirectoryName), "/fCG_"));

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

  fprintf(FP, "\t\"Nodes\": [\n");
  CGNode *CurrentNode = CG->NodesLinkedListHead;
  while (CurrentNode!=NULL) {
    fprintf(FP,
            "\t\t{\n"
            "\t\t\t\"NodeId\":%d,\n"
            "\t\t\t\"Instruction\":\"%s\",\n"
            "\t\t\t\"NodeKind\": %d,\n"
            "\t\t\t\"Height\": %d,\n"
            "\t\t\t\"RootNode\": %d,\n"
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

  printf("\nComputation Graph written to file: %s\n", FileName);
#endif
}

void fCGDotGraph() {
#if FAF_DEBUG
  // Create a directory if not present
  char *DirectoryName = (char *)malloc((strlen(LOG_DIRECTORY_NAME)+1) * sizeof(char));
  strcpy(DirectoryName, LOG_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  char ExecutionId[5000];
  char FileName[5000];
  FileName[0] = '\0';
  strcpy(FileName, strcat(strcpy(FileName, DirectoryName), "/fCGDot_"));

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

  printf("\nDot Graph written to file: %s\n", FileName);
#endif
}




#endif // LLVM_COMPUTATIONGRAPH_H
