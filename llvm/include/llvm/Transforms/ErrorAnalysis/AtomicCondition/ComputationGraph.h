//
// Created by tanmay on 6/21/22.
//

#ifndef LLVM_COMPUTATIONGRAPH_H
#define LLVM_COMPUTATIONGRAPH_H

#include "AtomicCondition.h"


/*----------------------------------------------------------------------------*/
/* Data Structures, Associated Functions and Types                            */
/*----------------------------------------------------------------------------*/

enum NodeKind {
  Register,
  UnaryInstruction,
  BinaryInstruction
};

struct CGNode {
  int NodeId;
  char *InstructionString;
  enum NodeKind Kind;
  struct CGNode *LeftNode;
  struct CGNode *RightNode;
  ACItem **ACRecord;
  int Height;
  int RootNode;
  char *FileName;
  int LineNumber;
};

typedef struct CGNode CGNode;

struct InstructionNodePair {
  char *InstructionString;
  CGNode **Node;
};

typedef struct InstructionNodePair InstructionNodePair;

struct PhiNode {
  char *PhiInstruction;
  char *ResidentInBB;
  int NumBranches;
  char **IncomingVals;
  char **BasicBlocks;
};

typedef struct PhiNode PhiNode;

struct ComputationGraph {
  uint64_t CGNodesListLength;
  CGNode **CGNodes;
  uint64_t InstructionNodeMapLength;
  InstructionNodePair **InstructionNodeMap;
  char **BasicBlockExecutionChainTail;
  PhiNode **PhiNodes;
  uint64_t PhiNodesListLength;
};

typedef struct ComputationGraph ComputationGraph;

/*----------------------------------------------------------------------------*/
/* Constants                                                                  */
/*----------------------------------------------------------------------------*/
#define LOG_DIRECTORY_NAME ".fAF_logs"
#define BASIC_BLOCK_EXECUTION_CHAIN_SIZE 1000000
#define CG_NODE_LIST_SIZE 1000000
#define INSTRUCTION_NODE_MAP_SIZE 100000
#define PHI_NODE_LIST_SIZE 100000

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/

ComputationGraph *CG;
int NodeCounter;

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
  assert(strlen(Value) != 0);

  // An LLVM register always begins with a '%' symbol
  if(Value[0] == '%')
    return 1;
  return 0;
}

int fCGisRegisterNamed(char *Register) {
  // A named LLVM register will have an alphabet for its 2nd character.
  // 1st character is always '%'.
  assert(fCGisRegister(Register));
  if(isdigit(Register[1]))
    return 0;
  return 1;
}

int fCGisPHIInstruction(char *InstructionString) {
  if(strstr(InstructionString, "phi") != NULL)
    return 1;
  return 0;
}

// Checks if this Phi Instruction was recorded.
int fCGPHIInstructionRecorded(char *Instruction) {
  assert(fCGisPHIInstruction(Instruction));

  // Return if Phi Nodes list is empty
  if(CG->PhiNodesListLength == 0)
    return 0;

  // Traverse the phi nodes list till record found
  PhiNode **CurrPhiNode = CG->PhiNodes;
  while(*CurrPhiNode != NULL && strcmp((*CurrPhiNode)->PhiInstruction, Instruction)!=0) {
    CurrPhiNode++;
  }

  // If record found return true
  if(*CurrPhiNode != NULL)
    return 1;

  // Record not found, return false
  return 0;
}

void fAFCopyCGNodePointerArray(CGNode ***Dest, CGNode ***Src, uint64_t SrcLength) {
  for (uint64_t I = 0, J = 0; I < SrcLength; ++I, ++J) {
    (*Dest)[J] = (*Src)[I];
  }
}

CGNode *fCGInstructionNodeMapGet(InstructionNodePair **InstructionNodeMap, char *Instruction) {
#if FAF_DEBUG>=2
  printf("\n\tSearching for Node corresponding to Instruction: %s\n",
         Instruction);
#endif
  InstructionNodePair **CurrPair = InstructionNodeMap;
  while (*CurrPair != NULL && strncmp((*CurrPair)->InstructionString,
                                        Instruction,
                                        strlen(Instruction)) != 0) {
    #if FAF_DEBUG>=2
    printf("\t\tCurrent Pair: %s->%d\n",
           (*CurrPair)->InstructionString,
           (*(*CurrPair)->Node)->NodeId);
    #endif
    CurrPair++;
  }
  assert(*CurrPair != NULL);
  return *(*CurrPair)->Node;
}

/*----------------------------------------------------------------------------*/
/* Primary Functions                                                          */
/*----------------------------------------------------------------------------*/

void fCGInitialize() {
  // Allocating memory for the graph itself
  if((CG = (ComputationGraph*)malloc(sizeof(ComputationGraph))) == NULL) {
    printf("#CG: Not enough memory for Computation Graph!");
    exit(EXIT_FAILURE);
  }

  // Allocate memory to the CGNodes
  if( (CG->CGNodes =
           (CGNode **)malloc(sizeof(CGNode *) *
                             CG_NODE_LIST_SIZE)) == NULL) {
    printf("#CG: Not enough memory for the CGNodes!");
    exit(EXIT_FAILURE);
  }
  CG->CGNodesListLength=0;

  // Allocate memory to the InstructionNodeMap
  if( (CG->InstructionNodeMap =
           (InstructionNodePair **)malloc(sizeof(InstructionNodePair *) *
                                          INSTRUCTION_NODE_MAP_SIZE)) == NULL) {
    printf("#CG: Not enough memory for the InstructionNodeMap!");
    exit(EXIT_FAILURE);
  }
  CG->InstructionNodeMapLength=0;

  // Allocate memory to the PhiNodes
  if( (CG->PhiNodes =
           (PhiNode **)malloc(sizeof(PhiNode *) *
                              PHI_NODE_LIST_SIZE)) == NULL) {
    printf("#CG: Not enough memory for the PhiNodes!");
    exit(EXIT_FAILURE);
  }
  CG->PhiNodesListLength=0;

  // Allocate memory to the Basic Block Execution Chain
  if( (CG->BasicBlockExecutionChainTail =
           (char **)malloc((size_t)((int64_t)sizeof(char*) *
                                    BASIC_BLOCK_EXECUTION_CHAIN_SIZE))) == NULL) {
    printf("#CG: Not enough memory for the BasicBlockExecutionChain!");
    exit(EXIT_FAILURE);
  }
  // The first pointer will never be assigned and will remain NULL acting as a
  // boundary/terminator for anything traversing the chain in reverse.
  char **BasicBlockPointer = CG->BasicBlockExecutionChainTail;
  for (int Index = 0; Index < BASIC_BLOCK_EXECUTION_CHAIN_SIZE; ++Index, BasicBlockPointer++) {
    *BasicBlockPointer = NULL;
  }

#if FAF_DEBUG
  // Generate a file path + file name
  char ACFile[5000];
  char CGFile[5000];
  char CGDotFile[5000];

  fAFGenerateFileString(ACFile, "fAC_", ".json");
  fAFGenerateFileString(CGFile, "fCG_", ".json");
  fAFGenerateFileString(CGDotFile, "fCGDot_", ".gv");

  printf("Atomic Conditions Storage File:%s\n", ACFile);
  printf("Computation Graph Storage File:%s\n", CGFile);
  printf("Dot Graph File:%s\n", CGDotFile);
#endif
}

// Add Phi node to Phi nodes list
void fCGrecordPHIInstruction(char *InstructionString, char *ResidentBBName) {
  assert(fCGisPHIInstruction(InstructionString));
  char *PhiInstruction = InstructionString;

  if(fCGPHIInstructionRecorded(PhiInstruction))
    return ;

#if FAF_DEBUG>=2
    printf("Recording PHI Instruction\n");
    printf("\tPHI Instruction String:%s\n", PhiInstruction);
#endif

  char **IncomingVals, **BasicBlocks;

  // This character pointer will be used to traverse the PhiInstruction
  // and copy the basic block names and values from it.
  char *CharFindingPointer = PhiInstruction;
  int NumIncomingBranches;

  // Counting number of incoming branches in PHI instruction
  for (NumIncomingBranches=0; CharFindingPointer[NumIncomingBranches];
       CharFindingPointer[NumIncomingBranches]=='[' ? NumIncomingBranches++ : *CharFindingPointer++);

  // Allocating Memory for character arrays
  IncomingVals = (char**)malloc ( (NumIncomingBranches) * sizeof (char*));
  BasicBlocks = (char**)malloc ( (NumIncomingBranches) * sizeof (char*));

  CharFindingPointer = PhiInstruction;
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

  PhiNode *Node = NULL;

  if((Node = (PhiNode *)malloc(sizeof(PhiNode))) == NULL) {
    printf("#fAC: PhiNodeList out of memory!");
    exit(EXIT_FAILURE);
  }

  Node->PhiInstruction = PhiInstruction;
  Node->ResidentInBB = ResidentBBName;
  Node->NumBranches = NumIncomingBranches;
  Node->IncomingVals = IncomingVals;
  Node->BasicBlocks = BasicBlocks;

  // Adding new PhiNode to PhiNodes list.
  CG->PhiNodes[CG->PhiNodesListLength] = Node;
  CG->PhiNodesListLength++;

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
char *fCGperformPHIResolution(char *Instruction) {
  if(!fCGisPHIInstruction(Instruction))
    return Instruction;

  char *PHIInstruction = Instruction;

#if FAF_DEBUG
  printf("\nPerforming PHI Instruction Resolution to a Non-PHI Instruction\n");
#endif

  // Copying PHI Instruction Register Name
  const unsigned long PhiInstructionLen = strlen(PHIInstruction);
  char *ResolvedValue = (char*)malloc ( (PhiInstructionLen+1) * sizeof (char));
  strncpy(ResolvedValue, PHIInstruction, PhiInstructionLen);
  ResolvedValue[PhiInstructionLen]=0;

#if FAF_DEBUG
  printf("\tFirst PHI Instruction: %s\n", ResolvedValue);
#endif

  // Setting Basic Block Execution trace pointer to the tail of Basic Block
  // Execution chain
  char *PreviousBasicBlock = *(CG->BasicBlockExecutionChainTail-1);

  // Finding Instruction record in Phi Nodes List
#if FAF_DEBUG>=2
  printf("\tSearching for record of FIRST PHI Instruction in PHI Nodes List\n");
#endif

  PhiNode **CurrPhiNode = CG->PhiNodes;
  while(*CurrPhiNode != NULL && strcmp((*CurrPhiNode)->PhiInstruction, ResolvedValue) != 0) {
#if FAF_DEBUG>=2
    printf("\t\tPHI Nodes List: Current PHI Instruction Register Name: %s\n",
       (*CurrPhiNode)->PhiInstruction);
#endif
    CurrPhiNode++;
  }
  // The PHI Instruction is already recorded before it is invoked.
  assert(*CurrPhiNode != NULL);
#if FAF_DEBUG>=2
  printf("\t\tRecord found! - %s\n", (*CurrPhiNode)->PhiInstruction);
#endif

  // Loop to traverse the PHI Chain in reverse order of execution till a Non-PHI
  // Instruction is found
#if FAF_DEBUG>=2
  printf("\n\tBackTracking Non-PHI Instruction\n");
#endif

  while (1) {
    int BranchCounter;

    //
    while(1) {
    // Setting Pointers to traverse PHI Node's basic blocks and corresponding
    // incoming values
    char **IncomingValsUnit = (*CurrPhiNode)->IncomingVals;
    char **BasicBlocksUnit = (*CurrPhiNode)->BasicBlocks;

    // Finding the Basic Block in the PHI Instruction record
#if FAF_DEBUG>=2
    printf("\t\tPrevious Basic Block:%s\n", PreviousBasicBlock);
#endif

      // Looping through the Phi Nodes branches till the PreviousBasicBlock
      // matches some basic block in the Phi Nodes branches.
      for (BranchCounter = 0;
           BranchCounter < (*CurrPhiNode)->NumBranches &&
           strcmp(*BasicBlocksUnit, PreviousBasicBlock) != 0;
           BranchCounter++, IncomingValsUnit++, BasicBlocksUnit++);

        // Debugging section printing the Basic Block Execution Chain
#if FAF_DEBUG >= 2
      char **BBPointer = CG->BasicBlockExecutionChainTail;
      printf("\n\t\tReversed Basic Block Execution Chain\n");
      while (*BBPointer != NULL) {
        printf("\t\t\tBasic Block Execution Chain: %s\n", *BBPointer);
        BBPointer--;
      }
      printf("\n");
#endif


      // If branch corresponding to previous basic block is not found jump
      // back another basic block in the execution chain.
      if(BranchCounter < (*CurrPhiNode)->NumBranches) {
        // Assuming the correct branch is chosen, saving the Incoming Register
        // and breaking out of the loop.
        ResolvedValue = *IncomingValsUnit;
        break;
      }
      PreviousBasicBlock--;
    }


#if FAF_DEBUG>=2
    printf("\t\tResolved Value:%s\n", ResolvedValue);
#endif

    // Resolved Value is not a register. Probably is a constant so return it.
    if(!fCGisRegister(ResolvedValue))
      return ResolvedValue;

    // Finding Instruction record in Phi Nodes List
#if FAF_DEBUG>=2
    printf("\t\tSearching for record of PHI Instruction in PHI Nodes List\n");
#endif
    CurrPhiNode = CG->PhiNodes;
    while(*CurrPhiNode != NULL && strncmp((*CurrPhiNode)->PhiInstruction,
                                          ResolvedValue, strlen(ResolvedValue)) != 0) {
#if FAF_DEBUG>=2
      printf("\t\t\tPHI Nodes List: Current PHI Instruction Register Name:%s\n",
       (*CurrPhiNode)->PhiInstruction);
#endif
      CurrPhiNode++;
    }

    // Phi node corresponding to ResolvedVal not found. Its a Register that does
    // not contain a Phi Instruction, break out of the loop.
    if(*CurrPhiNode == NULL) {
#if FAF_DEBUG>=2
      printf("\t\tPHI Node not found corresponding to %s\n", ResolvedValue);
#endif
      break;
    }

#if FAF_DEBUG>=2
    printf("\t\t\tRecord found! - %s\n", (*CurrPhiNode)->PhiInstruction);
#endif

    // Jumping one or more Basic Blocks in the Basic Block Execution Chain to
    // reach a Basic Block where the New Phi Node resides.
#if FAF_DEBUG>=2
    printf("\t\t%s resides in Basic Block %s\n", ResolvedValue,
           (*CurrPhiNode)->ResidentInBB);
#endif
    while(strcmp((*CurrPhiNode)->ResidentInBB, PreviousBasicBlock) != 0)
      PreviousBasicBlock--;
    PreviousBasicBlock--;
  }

#if FAF_DEBUG
  printf("\n\tLooking for Register %s in Nodes List.\n", ResolvedValue);
#endif

  CGNode **CurrNode = CG->CGNodes;
  // Finding instruction in Nodes List
  while (*CurrNode != NULL &&
         strncmp(ResolvedValue, (*CurrNode)->InstructionString,
                 strlen(ResolvedValue)) != 0) {
#if FAF_DEBUG
    printf("\t\tNodes List:%s\n", (*CurrNode)->InstructionString);
#endif
    CurrNode++;
  }

  if(*CurrNode != NULL)
    ResolvedValue = (*CurrNode)->InstructionString;
  else
    assert(!fCGisRegister(ResolvedValue));

#if FAF_DEBUG
  printf("\tFinal Resolved Value:%s\n\n", ResolvedValue);
#endif
  return ResolvedValue;
}

void fCGcreateNode(char *InstructionString, char *LeftOpInstructionString,
                   char *RightOpInstructionString, enum NodeKind NK,
                   ACItem **ACRecord, char *FileName, int LineNumber){
  // Ensuring Instruction to create node for is not a Phi node.
  assert(!fCGisPHIInstruction(InstructionString));

#if FAF_DEBUG
  printf("Creating New Node in Computation graph\n");
  printf("\tInstruction String: %s\n", InstructionString);
  printf("\tLeft Instruction String: %s\n", LeftOpInstructionString);
  printf("\tRight Instruction String: %s\n", RightOpInstructionString);
  printf("\tNode Kind: %d\n", NK);
  if(ACRecord != NULL)
    printf("\tAC Record Id: %d\n", (*ACRecord)->ItemId);
  printf("\tCurrent BasicBlockName: %s\n", *CG->BasicBlockExecutionChainTail);
  if (strcmp(*CG->BasicBlockExecutionChainTail, "%entry") != 0)
    printf("\tPrevious BasicBlockName: %s\n", *(CG->BasicBlockExecutionChainTail-1));
#endif
  InstructionNodePair **CurrPair=CG->InstructionNodeMap;


  // Allocating memory for new CGNode, FileName and new InstructionNodePair
  // Adding Node to array
  if((CG->CGNodes[CG->CGNodesListLength] = (CGNode *)malloc(sizeof(CGNode))) == NULL) {
    printf("#fAC: AC table out of memory error!");
    exit(EXIT_FAILURE);
  }
  CG->CGNodes[CG->CGNodesListLength]->NodeId = NodeCounter;
  CG->CGNodes[CG->CGNodesListLength]->InstructionString = InstructionString;
  CG->CGNodes[CG->CGNodesListLength]->Kind = NK;
  CG->CGNodes[CG->CGNodesListLength]->LeftNode = NULL;
  CG->CGNodes[CG->CGNodesListLength]->RightNode = NULL;
  CG->CGNodes[CG->CGNodesListLength]->ACRecord = ACRecord;
  CG->CGNodes[CG->CGNodesListLength]->Height = 0;
  CG->CGNodes[CG->CGNodesListLength]->RootNode = 1;
  CG->CGNodes[CG->CGNodesListLength]->FileName = FileName;
  CG->CGNodes[CG->CGNodesListLength]->LineNumber = LineNumber;


  // Update/Insert a New Key-Value pair in InstructionNodeMap
  while (*CurrPair != NULL &&
         (*CurrPair)->InstructionString != InstructionString)
    CurrPair++;

  if (*CurrPair != NULL) {
    (*CurrPair)->Node = &(CG->CGNodes[CG->CGNodesListLength]);
  } else { // Nope, couldn't find it
    if((CG->InstructionNodeMap[CG->InstructionNodeMapLength] =
             (InstructionNodePair *)malloc(sizeof(InstructionNodePair))) == NULL) {
      printf("#fAC: Not enough memory for a new InstructionNodePair!");
      exit(EXIT_FAILURE);
    }
    CG->InstructionNodeMap[CG->InstructionNodeMapLength]->InstructionString = InstructionString;
    CG->InstructionNodeMap[CG->InstructionNodeMapLength]->Node = &(CG->CGNodes[CG->CGNodesListLength]);
    CG->InstructionNodeMapLength++;
  }

  NodeCounter++;

  char *ResolvedLeftValue=fCGperformPHIResolution(LeftOpInstructionString);
  char *ResolvedRightValue=fCGperformPHIResolution(RightOpInstructionString);

  // Linking Left and Right operand nodes to Node if any
  switch (NK) {
  case 0:
#if FAF_DEBUG
    printf("\n\tIn Register case\n\n");
#endif
    break;
  case 1:
#if FAF_DEBUG
    printf("\n\tIn Unary case\n");
#endif
    // Setting the Left Node

    // If Resolved Value is an Instruction, find the corresponding Node in
    // CGNodes.
    if(strlen(ResolvedLeftValue) != 0 &&
        fCGisRegister(ResolvedLeftValue)) {
#if FAF_DEBUG>=2
      printf("\tGet Node Corresponding to ResolvedLeftValue: %s\n",
             ResolvedLeftValue);
#endif
      CG->CGNodes[CG->CGNodesListLength]->LeftNode =
          fCGInstructionNodeMapGet(CG->InstructionNodeMap,
                                   ResolvedLeftValue);
    }

    // If Left Node has been linked, update this nodes height correspondingly
    // and mark the Left Node as NOT a Root Node.
    if(CG->CGNodes[CG->CGNodesListLength]->LeftNode != NULL) {
      CG->CGNodes[CG->CGNodesListLength]->Height = CG->CGNodes[CG->CGNodesListLength]->LeftNode->Height;
      CG->CGNodes[CG->CGNodesListLength]->LeftNode->RootNode = 0;
    }
    CG->CGNodes[CG->CGNodesListLength]->Height = CG->CGNodes[CG->CGNodesListLength]->Height+1;
    CG->CGNodes[CG->CGNodesListLength]->RootNode = 1;

#if FAF_DEBUG>=2
    printf("\n");
#endif
    break;
  case 2:
#if FAF_DEBUG
    printf("\n\tIn Binary case\n");
#endif
    // Setting the Left Node

    // If Resolved Value is an Instruction, find the corresponding Node in
    // CGNodes.
    if(strlen(ResolvedLeftValue) != 0 &&
        fCGisRegister(ResolvedLeftValue)) {
#if FAF_DEBUG>=2
      printf("\tGet Node Corresponding to ResolvedLeftValue: %s\n",
             ResolvedLeftValue);
#endif
      CG->CGNodes[CG->CGNodesListLength]->LeftNode =
          fCGInstructionNodeMapGet(CG->InstructionNodeMap,
                                   ResolvedLeftValue);
    }

    // Setting the Right Node

    // If Resolved Value is an Instruction, find the corresponding Node in
    // CGNodes.
    if(strlen(ResolvedRightValue) != 0 &&
        fCGisRegister(ResolvedRightValue)) {
#if FAF_DEBUG>=2
      printf("\tGet Node Corresponding to ResolvedRightValue: %s\n",
             ResolvedRightValue);
#endif
      CG->CGNodes[CG->CGNodesListLength]->RightNode =
          fCGInstructionNodeMapGet(CG->InstructionNodeMap,
                                   ResolvedRightValue);
    }

    // If Left Node and/or Right Node have been linked, update this nodes height
    // correspondingly and mark the Left AND Right Nodes as NOT a Root Node.
    if(CG->CGNodes[CG->CGNodesListLength]->LeftNode != NULL) {
      CG->CGNodes[CG->CGNodesListLength]->Height =
          CG->CGNodes[CG->CGNodesListLength]->LeftNode->Height;
      CG->CGNodes[CG->CGNodesListLength]->LeftNode->RootNode = 0;
    }

    if(CG->CGNodes[CG->CGNodesListLength]->RightNode != NULL &&
        CG->CGNodes[CG->CGNodesListLength]->Height < CG->CGNodes[CG->CGNodesListLength]->RightNode->Height) {
      CG->CGNodes[CG->CGNodesListLength]->Height =
          CG->CGNodes[CG->CGNodesListLength]->RightNode->Height;
      CG->CGNodes[CG->CGNodesListLength]->RightNode->RootNode = 0;
    }

    CG->CGNodes[CG->CGNodesListLength]->Height = CG->CGNodes[CG->CGNodesListLength]->Height+1;
    CG->CGNodes[CG->CGNodesListLength]->RootNode = 1;

#if FAF_DEBUG>=2
    printf("\n");
#endif
    break;
  default:
    printf("#fAC: Node Kind Unknown!");
    exit(EXIT_FAILURE);
  }


#if FAF_DEBUG
  printf("\tNew Node Created\n");
  printf("\t\tNode ID: %d\n", CG->CGNodes[CG->CGNodesListLength]->NodeId);
  printf("\t\tInstruction: %s\n", CG->CGNodes[CG->CGNodesListLength]->InstructionString);
  printf("\t\tNode Type: %d\n", CG->CGNodes[CG->CGNodesListLength]->Kind);
  if(CG->CGNodes[CG->CGNodesListLength]->LeftNode != NULL)
    printf("\t\tResolved Left Instruction: %s\n",
           CG->CGNodes[CG->CGNodesListLength]->LeftNode->InstructionString);
  if(CG->CGNodes[CG->CGNodesListLength]->RightNode != NULL)
    printf("\t\tResolved Right Instruciton: %s\n",
           CG->CGNodes[CG->CGNodesListLength]->RightNode->InstructionString);
  if(ACRecord != NULL)
    printf("\t\tAC Record Id: %d\n", (*CG->CGNodes[CG->CGNodesListLength]->ACRecord)->ItemId);
  printf("\t\tNode Height: %d\n", CG->CGNodes[CG->CGNodesListLength]->Height);
  printf("\t\tIs a Root Node? %s\n", CG->CGNodes[CG->CGNodesListLength]->RootNode?"yes":"no");
  printf("\tNode Creation Completed\n\n");
#endif

  // New Node Added successfully. Update List Size
  CG->CGNodesListLength++;
  return ;
}

void fCGStoreCG() {
#if FAF_DEBUG
  // Generate a file path + file name string to store the Computation Graph nodes
  char File[5000];
  fAFGenerateFileString(File, "fCG_", ".json");

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

  fprintf(FP, "\t\"Nodes\": [\n");
  CGNode **CurrentNode = CG->CGNodes;
  while (*CurrentNode != NULL) {
    fprintf(FP,
            "\t\t{\n"
            "\t\t\t\"NodeId\":%d,\n"
            "\t\t\t\"Instruction\":\"%s\",\n"
            "\t\t\t\"Node Kind\": %d,\n"
            "\t\t\t\"Height\": %d,\n"
            "\t\t\t\"Root Node\": %d,\n"
            "\t\t\t\"Left Node\": %d,\n"
            "\t\t\t\"Right Node\": %d,\n"
            "\t\t\t\"AC Record Id\": %d,\n"
            "\t\t\t\"File Name\":\"%s\",\n"
            "\t\t\t\"Line Number\": %d\n",
            (*CurrentNode)->NodeId,
            (*CurrentNode)->InstructionString,
            (*CurrentNode)->Kind,
            (*CurrentNode)->Height,
            (*CurrentNode)->RootNode,
            ((*CurrentNode)->LeftNode != NULL?(*CurrentNode)->LeftNode->NodeId:-1),
            ((*CurrentNode)->RightNode != NULL?(*CurrentNode)->RightNode->NodeId:-1),
            ((*CurrentNode)->ACRecord != NULL?(*(*CurrentNode)->ACRecord)->ItemId:-1),
            (*CurrentNode)->FileName,
            (*CurrentNode)->LineNumber);

    if (*(CurrentNode+1) != NULL)
      fprintf(FP, "\t\t},\n");
    else
      fprintf(FP, "\t\t}\n");
    CurrentNode++;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");

  fclose(FP);

  printf("\nComputation Graph written to file: %s\n", File);
#endif
}

void fCGDotGraph() {
#if FAF_DEBUG
  // Generate a file path + file name string to for a Dot graph of the Computation
  // DAG
  char File[5000];
  fAFGenerateFileString(File, "fCGDot_", ".gv");

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
  FILE *FP = fopen(File, "w");
  fprintf(FP, "digraph ");
  fprintf(FP, "G ");
  fprintf(FP, "{\n");

  CGNode **CurrentNode = CG->CGNodes;
  while (*CurrentNode != NULL) {
    switch ((*CurrentNode)->Kind) {
    case 0:
      fprintf(FP, "\t%d [shape=rectangle];\n", (*CurrentNode)->NodeId);
      break;
    case 1:
      if ((*CurrentNode)->LeftNode != NULL)
        fprintf(FP, "\t%d -> %d;\n",
                (*CurrentNode)->LeftNode->NodeId,
                (*CurrentNode)->NodeId);
      break;
    case 2:
      if ((*CurrentNode)->LeftNode != NULL)
        fprintf(FP, "\t%d -> %d;\n",
                (*CurrentNode)->LeftNode->NodeId,
                (*CurrentNode)->NodeId);
      if ((*CurrentNode)->RightNode != NULL)
        fprintf(FP, "\t%d -> %d;\n",
                (*CurrentNode)->RightNode->NodeId,
                (*CurrentNode)->NodeId);
      break;
    default:
      break;
    }
    CurrentNode++;
  }

  // Creating Legend
  fprintf(FP, "\tsubgraph cluster {\n");
  fprintf(FP, "\t\tnode [shape=plaintext];\n");
  fprintf(FP, "\t\tlabel = \"Legend\";\n");
  fprintf(FP, "\t\tkey [label=<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\">\n");
  CurrentNode = CG->CGNodes;
  while (*CurrentNode != NULL) {
    fprintf(FP, "\t\t\t<tr><td>%d</td><td align=\"left\">%s</td></tr>\n",
            (*CurrentNode)->NodeId, (*CurrentNode)->InstructionString);
    CurrentNode++;
  }
  fprintf(FP, "\t\t\t</table>>]\n");

  // Ending Legend
  fprintf(FP, "\t}\n");

  // Ending Digraph
  fprintf(FP, "}\n");

  fclose(FP);

  printf("\nDot Graph written to file: %s\n", File);
#endif
}

#endif // LLVM_COMPUTATIONGRAPH_H