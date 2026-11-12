// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// G_AND is commutable and $x is defined by another instruction, so the rule
// is also emitted with the G_AND operands swapped.
def NestedDef : GICombineRule<
  (defs root:$dst),
  (match (G_XOR $x, $a, $b),
         (G_AND $dst, $x, $y)),
  (apply (G_OR $dst, $a, $y))>;

// Neither operand is constrained: the swapped version can never be used.
def FreeOperands : GICombineRule<
  (defs root:$dst),
  (match (G_AND $dst, $x, $y)),
  (apply (G_OR $dst, $x, $y))>;

// Both operands are the same register: nothing to swap.
def SameOperand : GICombineRule<
  (defs root:$dst),
  (match (G_AND $dst, $x, $x), "return isFoo(${x});"),
  (apply (GIReplaceReg $dst, $x))>;

// G_SUB is not commutable.
def NotCommutable : GICombineRule<
  (defs root:$dst),
  (match (G_XOR $x, $a, $b),
         (G_SUB $dst, $x, $y)),
  (apply (G_ADD $dst, $a, $y))>;

// An immediate operand is constrained.
def ImmOperand : GICombineRule<
  (defs root:$dst),
  (match (G_MUL $dst, $x, 0)),
  (apply (G_CONSTANT $dst, (i32 0)))>;

// An operand referenced from C++ code is constrained.
def CXXOperand : GICombineRule<
  (defs root:$dst),
  (match (G_ADD $dst, $x, $y), "return isFoo(${y});"),
  (apply (GIReplaceReg $dst, $x))>;

// Each commutable instruction doubles the permutations: 2 x 2 = 4.
def TwoCommutable : GICombineRule<
  (defs root:$dst),
  (match (G_XOR $x, $a, $b),
         (G_ADD $y, $c, $x),
         (G_AND $dst, $y, $d)),
  (apply (GIReplaceReg $dst, $a))>;

// A commutable target instruction is not commuted: the target decides which
// of its operands commute.
def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
def MY_ADD : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = (outs GPR32:$dst);
  let InOperandList = (ins GPR32:$src0, GPR32:$src1);
  let isCommutable = 1;
}

def TargetInst : GICombineRule<
  (defs root:$dst),
  (match (G_XOR $x, $a, $b),
         (MY_ADD $dst, $x, $y)),
  (apply (G_OR $dst, $a, $y))>;

// Commutable instructions in a PatFrag alternative are commuted separately
// for each use of the PatFrag. The shared $p constrains both uses.
def XorFrag : GICombinePatFrag<
  (outs root:$dst), (ins gi_mo:$a),
  [(pattern (G_XOR $dst, $a, $b))]>;

def PatFragUses : GICombineRule<
  (defs root:$dst),
  (match (XorFrag $x, $p):$f0,
         (XorFrag $y, $p):$f1,
         (G_SUB $dst, $x, $y)),
  (apply (GIReplaceReg $dst, $p))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  NestedDef,
  FreeOperands,
  SameOperand,
  NotCommutable,
  ImmOperand,
  CXXOperand,
  TwoCommutable,
  PatFragUses,
  TargetInst
]>;

// CHECK:      bool GenMyCombiner::testMIPredicate_MI(unsigned PredicateID, const MachineInstr & MI, const MatcherState &State) const {
// CHECK-NEXT:   switch (PredicateID) {
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner0: {
// CHECK-NEXT:     return isFoo(State.MIs[0]->getOperand(1));
// CHECK:        case GICXXPred_MI_Predicate_GICombiner1: {
// CHECK-NEXT:     return isFoo(State.MIs[0]->getOperand(2));

// CHECK: const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 7 //
// CHECK: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/0, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #5: CXXOperand
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 8 //
// CHECK: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/0, /*NewOpIdx*/2,
// CHECK: // Combiner Rule #5: CXXOperand @ [__CXXOperand_match_0[commuted]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 13 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/1, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #7: PatFragUses @ [f0[0], f1[0]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 14 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/2,
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/1, /*NewOpIdx*/2,
// CHECK: // Combiner Rule #7: PatFragUses @ [f0.__XorFrag_alt0_pattern_0[commuted], f0[0], f1[0]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 15 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/1, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #7: PatFragUses @ [f0[0], f1.__XorFrag_alt0_pattern_0[commuted], f1[0]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 16 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/1, /*NewOpIdx*/2,
// CHECK: // Combiner Rule #7: PatFragUses @ [f0.__XorFrag_alt0_pattern_0[commuted], f0[0], f1.__XorFrag_alt0_pattern_0[commuted], f1[0]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 4 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: // Combiner Rule #3: NotCommutable
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 5 //
// CHECK: GIM_CheckConstantInt8, /*MI*/0, /*Op*/2, 0,
// CHECK: // Combiner Rule #4: ImmOperand
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 6 //
// CHECK: GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK: // Combiner Rule #4: ImmOperand @ [__ImmOperand_match_0[commuted]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 3 //
// CHECK: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/0, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #2: SameOperand
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 9 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/2, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #6: TwoCommutable
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 10 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/2, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #6: TwoCommutable @ [__TwoCommutable_match_1[commuted]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 11 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/2, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #6: TwoCommutable @ [__TwoCommutable_match_2[commuted]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 12 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/2, /*NewOpIdx*/1,
// CHECK: // Combiner Rule #6: TwoCommutable @ [__TwoCommutable_match_1[commuted], __TwoCommutable_match_2[commuted]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 0 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: // Combiner Rule #0: NestedDef
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 1 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: // Combiner Rule #0: NestedDef @ [__NestedDef_match_1[commuted]]
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 2 //
// CHECK: // Combiner Rule #1: FreeOperands
// CHECK: GIM_Try, /*On fail goto*//*Label {{[0-9]+}}*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 17 //
// CHECK: GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_XOR),
// CHECK: // Combiner Rule #8: TargetInst
// CHECK-NOT: GIM_Try
// CHECK: GIM_Reject
