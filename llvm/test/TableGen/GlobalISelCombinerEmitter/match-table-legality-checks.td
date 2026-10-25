// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// A C++ apply with an element type, as in #231254.
def ShuffleOfBuildVector : GICombineRule<
  (defs root:$root),
  (match (G_BUILD_VECTOR $src1, GIVariadic<>:$unused),
         (G_IMPLICIT_DEF $src2),
         (G_SHUFFLE_VECTOR $dst, $src1, $src2, $mask):$root),
  (apply "APPLY")> {
  let LegalityChecks = [(G_BUILD_VECTOR $dst, (GIElementTypeOf $dst))];
}

// Several checks in one rule, including a single-type opcode.
def ZextOfTrunc : GICombineRule<
  (defs root:$dst),
  (match (G_TRUNC $t, $x),
         (G_ZEXT $dst, $t)),
  (apply "APPLY")> {
  let LegalityChecks = [(G_AND $dst), (G_ZEXT $dst, $x)];
}

// Without LegalityChecks, no legality predicate is generated.
def NoChecks : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $x)),
  (apply "APPLY")>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  ShuffleOfBuildVector,
  ZextOfTrunc,
  NoChecks
]>;
// CHECK:      bool GenMyCombiner::testMIPredicate_MI(unsigned PredicateID, const MachineInstr & MI, const MatcherState &State) const {
// CHECK-NEXT:   switch (PredicateID) {
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner0: {
// CHECK-NEXT:     return Helper.isLegalOrBeforeLegalizer({TargetOpcode::G_BUILD_VECTOR, {MRI.getType(State.MIs[0]->getOperand(0).getReg()), MRI.getType(State.MIs[0]->getOperand(0).getReg()).getElementType()}});
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner1: {
// CHECK-NEXT:     return Helper.isLegalOrBeforeLegalizer({TargetOpcode::G_AND, {MRI.getType(State.MIs[0]->getOperand(0).getReg())}});
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner2: {
// CHECK-NEXT:     return Helper.isLegalOrBeforeLegalizer({TargetOpcode::G_ZEXT, {MRI.getType(State.MIs[0]->getOperand(0).getReg()), MRI.getType(State.MIs[1]->getOperand(1).getReg())}});
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown predicate");