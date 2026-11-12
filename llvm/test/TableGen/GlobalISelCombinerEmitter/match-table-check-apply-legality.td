// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// Types come straight from the G_ANYEXT built by the apply: type0 from $root,
// type1 from $x.
def MergeUndefToAnyExt : GICombineRule<
  (defs root:$root),
  (match (G_IMPLICIT_DEF $undef),
         (G_MERGE_VALUES $root, $x, $undef)),
  (apply (G_ANYEXT $root, $x))> {
  let CheckApplyLegality = true;
}

// Shifts have two type indices: type0 from $d (the first operand of that
// type defined by the match) and type1 from the shift amount $amt.
def BitreverseShlToLShr : GICombineRule<
  (defs root:$d),
  (match (G_BITREVERSE $rev, $val),
         (G_SHL $src, $rev, $amt),
         (G_BITREVERSE $d, $src)),
  (apply (G_LSHR $d, $val, $amt))> {
  let CheckApplyLegality = true;
}

// Without CheckApplyLegality, no legality predicate is generated.
def MergeUndefNoCheck : GICombineRule<
  (defs root:$root),
  (match (G_IMPLICIT_DEF $undef),
         (G_MERGE_VALUES $root, $x, $undef)),
  (apply (G_ZEXT $root, $x))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  MergeUndefToAnyExt,
  BitreverseShlToLShr,
  MergeUndefNoCheck
]>;
// CHECK:      bool GenMyCombiner::testMIPredicate_MI(unsigned PredicateID, const MachineInstr & MI, const MatcherState &State) const {
// CHECK-NEXT:   switch (PredicateID) {
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner0: {
// CHECK-NEXT:     return Helper.isLegalOrBeforeLegalizer({TargetOpcode::G_ANYEXT, {MRI.getType(State.MIs[0]->getOperand(0).getReg()), MRI.getType(State.MIs[0]->getOperand(1).getReg())}});
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner1: {
// CHECK-NEXT:     return Helper.isLegalOrBeforeLegalizer({TargetOpcode::G_LSHR, {MRI.getType(State.MIs[0]->getOperand(0).getReg()), MRI.getType(State.MIs[1]->getOperand(2).getReg())}});
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown predicate");