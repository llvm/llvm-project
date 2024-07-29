// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -optimize-match-table=false -I %p/../../include -I %p/Common %s -o - | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def int_mytarget_sleep : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def G_TGT_CAT : MyTargetGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src0, untyped_imm_0:$immfield);
}

def TgtCat : SDNode<"MyTgt::CAT", SDTIntBinOp>;
def : GINodeEquiv<G_TGT_CAT, TgtCat>;


def SLEEP0 : I<(outs), (ins), []>;
def SLEEP1 : I<(outs), (ins), []>;
def CAT0 : I<(outs GPR32:$dst), (ins GPR32:$src0), []>;
def CAT1 : I<(outs GPR32:$dst), (ins GPR32:$src0), []>;

// Test immarg intrinsic pattern

// Make sure there is no type check.
// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// GISEL: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, GIMT_Encode2(Intrinsic::mytarget_sleep),
// GISEL-NEXT: // MIs[0] Operand 1
// GISEL-NEXT: GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
def : Pat<
  (int_mytarget_sleep 0),
  (SLEEP0)
>;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// GISEL: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, GIMT_Encode2(Intrinsic::mytarget_sleep),
// GISEL-NEXT: // MIs[0] Operand 1
// GISEL-NEXT: GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(1),
def : Pat<
  (int_mytarget_sleep 1),
  (SLEEP1)
>;

// Check a non-intrinsic instruction with an immediate parameter.

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(MyTarget::G_TGT_CAT),
// GISEL: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// GISEL-NEXT: // MIs[0] Operand 2
// GISEL-NEXT: GIM_CheckLiteralInt, /*MI*/0, /*Op*/2, GIMT_Encode8(0),
def : Pat<
  (TgtCat i32:$src0, 0),
  (CAT0 GPR32:$src0)
>;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(MyTarget::G_TGT_CAT),
// GISEL: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// GISEL-NEXT: // MIs[0] Operand 2
// GISEL-NEXT: GIM_CheckLiteralInt, /*MI*/0, /*Op*/2, GIMT_Encode8(93),
def : Pat<
  (TgtCat i32:$src0, 93),
  (CAT1 GPR32:$src0)
>;
