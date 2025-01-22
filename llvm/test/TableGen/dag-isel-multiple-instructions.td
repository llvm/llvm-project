// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def REG : Register<"REG">;
def GPR : RegisterClass<"TestTarget", [i32], 32, (add REG)>;


def FENCE : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
}

def LOAD : Instruction {
  let OutOperandList = (outs GPR:$r0);
  let InOperandList = (ins GPR:$t0);
  let mayLoad = 1;
}

def STORE : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins GPR:$t0,  GPR:$t1);
  let mayStore = 1;
}

class BINOP : Instruction {
  let OutOperandList = (outs GPR:$r0);
  let InOperandList = (ins GPR:$t0,  GPR:$t1);
}
def ADD : BINOP;
def SUB : BINOP;
def MUL : BINOP;
def DIV : BINOP;

def UDIVREM : Instruction {
  let OutOperandList = (outs GPR:$r1, GPR:$r0);
  let InOperandList = (ins GPR:$t0, GPR:$t1);
}

// The patterns are written here in the order that the emitter sorts them, so
// that the CHECK lines can stay next to them.

// CHECK-LABEL: static const unsigned char MatcherTable[] = {
// CHECK-NEXT: /*     0*/ OPC_SwitchOpcode /*5 cases */, 57, TARGET_VAL(ISD::STORE),// ->61
// CHECK-NEXT: /*     4*/  OPC_RecordMemRef,
// CHECK-NEXT: /*     5*/  OPC_RecordNode, // #0 = 'st' chained node
// CHECK-NEXT: /*     6*/  OPC_RecordChild1, // #1 = $val
// CHECK-NEXT: /*     7*/  OPC_CheckChild1TypeI32,
// CHECK-NEXT: /*     8*/  OPC_MoveChild2,
// CHECK-NEXT: /*     9*/  OPC_CheckOpcode, TARGET_VAL(ISD::LOAD),
// CHECK-NEXT: /*    12*/  OPC_RecordMemRef,
// CHECK-NEXT: /*    13*/  OPC_RecordNode, // #2 = 'ld' chained node
// CHECK-NEXT: /*    14*/  OPC_CheckFoldableChainNode,
// CHECK-NEXT: /*    15*/  OPC_MoveChild1,
// CHECK-NEXT: /*    16*/  OPC_CheckOpcode, TARGET_VAL(ISD::ADD),
// CHECK-NEXT: /*    19*/  OPC_RecordChild0, // #3 = $ptr
// CHECK-NEXT: /*    20*/  OPC_RecordChild1, // #4 = $offset
// CHECK-NEXT: /*    21*/  OPC_MoveParent,
// CHECK-NEXT: /*    22*/  OPC_CheckPredicate0,  // Predicate_unindexedload
// CHECK-NEXT: /*    23*/  OPC_CheckPredicate1,  // Predicate_load
// CHECK-NEXT: /*    24*/  OPC_MoveParent,
// CHECK-NEXT: /*    25*/  OPC_CheckPredicate2,  // Predicate_unindexedstore
// CHECK-NEXT: /*    26*/  OPC_CheckPredicate3,  // Predicate_store
// CHECK-NEXT: /*    27*/  OPC_EmitMergeInputChains, 2, 0, 2,
// CHECK-NEXT: /*    31*/  OPC_EmitNode0, TARGET_VAL(::FENCE), 0|OPFL_Chain|OPFL_MemRefs,
// CHECK-NEXT:                 0/*#Ops*/,
// CHECK-NEXT: /*    36*/  OPC_EmitNode1Chain, TARGET_VAL(::ADD),
// CHECK-NEXT:                 MVT::i32, 2/*#Ops*/, 3, 4,  // Results = #5
// CHECK-NEXT: /*    43*/  OPC_EmitNode1Chain, TARGET_VAL(::LOAD),
// CHECK-NEXT:                 MVT::i32, 1/*#Ops*/, 5,  // Results = #6
// CHECK-NEXT: /*    49*/  OPC_EmitNode0, TARGET_VAL(::STORE), 0|OPFL_Chain|OPFL_MemRefs,
// CHECK-NEXT:                 2/*#Ops*/, 1, 6,
// CHECK-NEXT: /*    56*/  OPC_MorphNodeTo0, TARGET_VAL(::FENCE), 0|OPFL_Chain|OPFL_MemRefs,
// CHECK-NEXT:                 0/*#Ops*/,
// CHECK-NEXT:             // Src: (st i32:{ *:[i32] }:$val, (ld:{ *:[iPTR] } (add:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$offset))<<P:Predicate_unindexedload>><<P:Predicate_load>>)<<P:Predicate_unindexedstore>><<P:Predicate_store>> - Complexity = 11
// CHECK-NEXT:             // Dst: ONE_TO_MANY: [
// CHECK-NEXT: 	(FENCE)
// CHECK-NEXT: 	(STORE i32:{ *:[i32] }:$val, (LOAD:{ *:[i32] } (ADD:{ *:[i32] } i32:{ *:[i32] }:$ptr, i32:{ *:[i32] }:$offset)))
// CHECK-NEXT: 	(FENCE)
// CHECK-NEXT: ]
def ONE_TO_MANY : Pattern<(store i32:$val, (load (add i32:$ptr, i32:$offset))),
  [
    (FENCE),
    (STORE i32:$val, (LOAD (ADD i32:$ptr, i32:$offset))),
    (FENCE)
  ]>;


// CHECK-EMPTY:
// CHECK-NEXT: /*    61*/ /*SwitchOpcode*/ 21, TARGET_VAL(ISD::CALLSEQ_START),// ->85
// CHECK-NEXT: /*    64*/  OPC_RecordNode, // #0 = 'AArch64callseq_start' chained node
// CHECK-NEXT: /*    65*/  OPC_RecordChild1, // #1 = $amt1
// CHECK-NEXT: /*    66*/  OPC_MoveChild1,
// CHECK-NEXT: /*    67*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
// CHECK-NEXT: /*    70*/  OPC_MoveSibling2,
// CHECK-NEXT: /*    71*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
// CHECK-NEXT: /*    74*/  OPC_RecordNode, // #2 = $amt2
// CHECK-NEXT: /*    75*/  OPC_MoveParent,
// CHECK-NEXT: /*    76*/  OPC_EmitMergeInputChains1_0,
// CHECK-NEXT: /*    77*/  OPC_MorphNodeTo1, TARGET_VAL(::ADJCALLSTACKDOWN), 0|OPFL_Chain|OPFL_GlueOutput,
// CHECK-NEXT:                 MVT::i32, 2/*#Ops*/, 1, 2,
// CHECK-NEXT:             // Src: (AArch64callseq_start (timm:{ *:[i32] }):$amt1, (timm:{ *:[i32] }):$amt2) - Complexity = 9
// CHECK-NEXT:             // Dst: ADJCALLSTACKDOWN: (ADJCALLSTACKDOWN:{ *:[i32] } (timm:{ *:[i32] }):$amt1, (timm:{ *:[i32] }):$amt2)
// Instruction with an implicit result (REG) in output pattern
def AArch64callseq_start : SDNode<"ISD::CALLSEQ_START",
                                SDCallSeqStart<[ SDTCisVT<0, i32>, SDTCisVT<1, i32> ]>,
                                [SDNPHasChain, SDNPOutGlue]>;
def ADJCALLSTACKDOWN : Instruction {
  dag OutOperandList = (outs);
  dag InOperandList  = (ins i32imm:$amt1, i32imm:$amt2);
  let Pattern        = [(AArch64callseq_start timm:$amt1, timm:$amt2)];
  let Defs = [REG];
}


// CHECK-EMPTY:
// CHECK-NEXT: /*    85*/ /*SwitchOpcode*/ 43, TARGET_VAL(ISD::UDIVREM),// ->131
// CHECK-NEXT: /*    88*/  OPC_RecordChild0, // #0 = $t0
// CHECK-NEXT: /*    89*/  OPC_RecordChild1, // #1 = $t1
// CHECK-NEXT: /*    90*/  OPC_Scope, 12, /*->104*/ // 2 children in Scope
// CHECK-NEXT: /*    92*/   OPC_EmitNode2None, TARGET_VAL(::INSTR),
// CHECK-NEXT:                  MVT::i32, MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2 #3
// CHECK-NEXT: /*   100*/   OPC_CompleteMatch, 2, 3, 2,
// CHECK-NEXT:               // Src: (udivrem:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1) - Complexity = 3
// CHECK-NEXT:               // Dst: INSTR: (INSTR:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1)
// An instruction with a pattern
def INSTR : Instruction {
  let OutOperandList = (outs GPR:$r1, GPR:$r0);
  let InOperandList = (ins GPR:$t0, GPR:$t1);
  let Pattern = [(set i32:$r0, i32:$r1, (udivrem i32:$t0, i32:$t1))];
}

// CHECK-EMPTY:
// CHECK-NEXT: /*   104*/  /*Scope*/ 25, /*->130*/
// CHECK-NEXT: /*   105*/   OPC_EmitNode0Chain, TARGET_VAL(::FENCE),
// CHECK-NEXT:                  0/*#Ops*/,
// CHECK-NEXT: /*   109*/   OPC_EmitNode1Chain, TARGET_VAL(::DIV),
// CHECK-NEXT:                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
// CHECK-NEXT: /*   116*/   OPC_EmitNode0Chain, TARGET_VAL(::FENCE),
// CHECK-NEXT:                  0/*#Ops*/,
// CHECK-NEXT: /*   120*/   OPC_EmitNode1Chain, TARGET_VAL(::LOAD),
// CHECK-NEXT:                  MVT::i32, 1/*#Ops*/, 0,  // Results = #3
// CHECK-NEXT: /*   126*/   OPC_CompleteMatch, 2, 2, 3,
// CHECK-NEXT:               // Src: (udivrem:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1) - Complexity = 3
// CHECK-NEXT:               // Dst: COLLECT_RESULTS: [
// CHECK-NEXT: 	(FENCE)
// CHECK-NEXT: 	(DIV:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1)
// CHECK-NEXT: 	(FENCE)
// CHECK-NEXT: 	(LOAD:{ *:[i32] } i32:{ *:[i32] }:$t0)
// CHECK-NEXT: ]
def COLLECT_RESULTS : Pattern<(udivrem i32:$t0, i32:$t1),
  [
    (FENCE),
    (DIV i32:$t0, i32:$t1),
    (FENCE),
    (LOAD i32:$t0),
  ]>;


// CHECK-EMPTY:
// CHECK-NEXT: /*   130*/  0, /*End of Scope*/
// CHECK-NEXT: /*   131*/ /*SwitchOpcode*/ 9, TARGET_VAL(ISD::SUB),// ->143
// CHECK-NEXT: /*   134*/  OPC_RecordChild0, // #0 = $t0
// CHECK-NEXT: /*   135*/  OPC_RecordChild1, // #1 = $t1
// CHECK-NEXT: /*   136*/  OPC_MorphNodeTo1None, TARGET_VAL(::SUB),
// CHECK-NEXT:                 MVT::i32, 2/*#Ops*/, 0, 1,
// CHECK-NEXT:             // Src: (sub:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1) - Complexity = 3
// CHECK-NEXT:             // Dst: ONE_TO_ONE: (SUB:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1)
def ONE_TO_ONE : Pattern<(sub i32:$t0, i32:$t1),
  [
    (SUB i32:$t0, i32:$t1)
  ]>;


// CHECK-EMPTY:
// CHECK-NEXT: /*   143*/ /*SwitchOpcode*/ 16, TARGET_VAL(ISD::MUL),// ->162
// CHECK-NEXT: /*   146*/  OPC_RecordChild0, // #0 = $t0
// CHECK-NEXT: /*   147*/  OPC_RecordChild1, // #1 = $t1
// CHECK-NEXT: /*   148*/  OPC_EmitNode1Chain, TARGET_VAL(::MUL),
// CHECK-NEXT:                 MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
// CHECK-NEXT: /*   155*/  OPC_EmitNode0Chain, TARGET_VAL(::FENCE),
// CHECK-NEXT:                 0/*#Ops*/,
// CHECK-NEXT: /*   159*/  OPC_CompleteMatch, 1, 2,
// CHECK-NEXT:              // Src: (mul:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1) - Complexity = 3
// CHECK-NEXT:              // Dst: ONE_TO_TWO: [
// CHECK-NEXT: 	(MUL:{ *:[i32] } i32:{ *:[i32] }:$t0, i32:{ *:[i32] }:$t1)
// CHECK-NEXT: 	(FENCE)
// CHECK-NEXT: ]
def ONE_TO_TWO : Pattern<(mul i32:$t0, i32:$t1),
  [
    (MUL i32:$t0, i32:$t1),
    (FENCE)
  ]>;

// CHECK-EMPTY:
// CHECK-NEXT: /*   162*/ 0, // EndSwitchOpcode
// CHECK-NEXT:     0
// CHECK-NEXT:   }; // Total Array size is 164 bytes
