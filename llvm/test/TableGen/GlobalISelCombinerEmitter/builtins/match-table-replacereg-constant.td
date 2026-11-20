// RUN: llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s --implicit-check-not=GIR_MakeTempReg \
// RUN:     --implicit-check-not=GIR_ReplaceRegWithTempReg

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// Each def of the root is rebuilt in place at its own operand index.
def ReplaceWithConstant : GICombineRule<
  (defs root:$dst),
  (match (G_UADDO $dst, $carry, $x, 0)),
  (apply (GIReplaceRegWithConstant $dst, -1),
         (GIReplaceRegWithConstant $carry, 0))>;

// The immediate is a raw IEEE bit pattern (single-precision pi/4).
def ReplaceWithFConstant : GICombineRule<
  (defs root:$dst),
  (match (G_FPEXT $dst, 0)),
  (apply (GIReplaceRegWithFConstant $dst, 0x3f490fdb))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  ReplaceWithConstant,
  ReplaceWithFConstant
]>;

// CHECK: // Combiner Rule #0: ReplaceWithConstant
// CHECK-NEXT: GIR_BuildConstantInPlace, /*InsnID*/0, /*OpIdx*/0, /*Val*/GIMT_Encode8(18446744073709551615u),
// CHECK-NEXT: GIR_BuildConstantInPlace, /*InsnID*/0, /*OpIdx*/1, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK: // Combiner Rule #1: ReplaceWithFConstant
// CHECK-NEXT: GIR_BuildFConstantInPlace, /*InsnID*/0, /*OpIdx*/0, /*Val*/GIMT_Encode8(1061752795),
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
