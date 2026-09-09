// RUN: llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def ReplaceWithFConstant : GICombineRule<
  (defs root:$dst),
  (match (G_UITOFP $dst, 0)),
  (apply (GIReplaceRegWithFConstant $dst, 0))>;

// Edge cases: the immediate is a raw IEEE bit pattern, not a decimal fp
// value, so this must survive as -1 (all-ones), not get reinterpreted.
def ReplaceWithFConstantAllOnes : GICombineRule<
  (defs root:$dst),
  (match (G_SITOFP $dst, 0)),
  (apply (GIReplaceRegWithFConstant $dst, -1))>;

// Arbitrary non-zero bit pattern (single-precision pi/4).
def ReplaceWithFConstantArbitrary : GICombineRule<
  (defs root:$dst),
  (match (G_FPEXT $dst, 0)),
  (apply (GIReplaceRegWithFConstant $dst, 0x3f490fdb))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  ReplaceWithFConstant,
  ReplaceWithFConstantAllOnes,
  ReplaceWithFConstantArbitrary
]>;

// CHECK: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/
// CHECK-NEXT: GIR_BuildFConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(1061752795),
// CHECK-NEXT: // Combiner Rule #2: ReplaceWithFConstantArbitrary
// CHECK-NEXT: GIR_ReplaceRegWithTempReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/
// CHECK-NEXT: GIR_BuildFConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(18446744073709551615u),
// CHECK-NEXT: // Combiner Rule #1: ReplaceWithFConstantAllOnes
// CHECK-NEXT: GIR_ReplaceRegWithTempReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/
// CHECK-NEXT: GIR_BuildFConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT: // Combiner Rule #0: ReplaceWithFConstant
// CHECK-NEXT: GIR_ReplaceRegWithTempReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
