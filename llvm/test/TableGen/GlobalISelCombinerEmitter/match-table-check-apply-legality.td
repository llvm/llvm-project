// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s --implicit-check-not=GIM_CheckCxxInsnPredicate \
// RUN:     --implicit-check-not=GIM_CheckIsLegal

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
// CHECK:      bool GenMyCombiner::isLegalForApply(unsigned Opcode, ArrayRef<LLT> Types) const {
// CHECK-NEXT:   return Helper.isLegalOrBeforeLegalizer(Opcode, Types);

// The legality checks are in the match table: each type index takes the type
// of a matched operand (MI, Op).
// CHECK: GIM_CheckIsLegal, GIMT_Encode2(TargetOpcode::G_ANYEXT), /*NumTypes*/2, /*MI*/0, /*Op*/0, /*MI*/0, /*Op*/1,
// CHECK: GIM_CheckIsLegal, GIMT_Encode2(TargetOpcode::G_LSHR), /*NumTypes*/2, /*MI*/0, /*Op*/0, /*MI*/1, /*Op*/2,
