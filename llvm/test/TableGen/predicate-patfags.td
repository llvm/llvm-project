// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common %s 2>&1 | FileCheck -check-prefixes=SDAG,SCUSTOM %s
// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common %s -DHASONEUSE 2>&1 | FileCheck -check-prefixes=SDAG,SBUILTIN %s
// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s 2>&1 | FileCheck -check-prefixes=GISEL,GCUSTOM %s
// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s -DHASONEUSE 2>&1 | FileCheck -check-prefixes=GISEL,GBUILTIN %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Test that a predicate works when there are multiple pattern trees
// in a PatFrags.

def int_tgt_mul24 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;
def int_tgt_mul24_2 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;

def TGTmul24_impl : SDNode<"TargetISD::MUL24", SDTIntBinOp>;

def TGTmul24 : PatFrags<(ops node:$src0, node:$src1),
  [(int_tgt_mul24 node:$src0, node:$src1),
   (TGTmul24_impl node:$src0, node:$src1)]>;


def G_TGT_MUL24 : GenericInstruction {
  let Namespace = "MyTarget";
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}


def : GINodeEquiv<G_TGT_MUL24, TGTmul24_impl>;


def TGTmul24_oneuse : PatFrag<
  (ops node:$src0, node:$src1),
  (TGTmul24 $src0, $src1)
#ifndef HASONEUSE
  , [{ return N->hasOneUse(); }]> {
  let GISelPredicateCode = [{
    return MRI->hasOneNonDBGUse(MI.getOperand(0).getReg());
  }];
#else
  > {
  let HasOneUse = 1;
#endif
}

// SDAG: OPC_CheckOpcode, TARGET_VAL(ISD::INTRINSIC_W_CHAIN),
// SDAG: OPC_CheckPredicate0, // Predicate_TGTmul24_oneuse

// SDAG: OPC_CheckOpcode, TARGET_VAL(TargetISD::MUL24),
// SDAG: OPC_CheckPredicate0, // Predicate_TGTmul24_oneuse

// SCUSTOM: return N->hasOneUse();
// SBUILTIN: if (!SDValue(N, 0).hasOneUse()) return false;

// GISEL: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// GISEL: GIM_CheckIntrinsicID, /*MI*/1, /*Op*/1, GIMT_Encode2(Intrinsic::tgt_mul24),
// GBUILTIN: GIM_CheckHasOneUse, /*MI*/1,
// GCUSTOM: GIM_CheckCxxInsnPredicate, /*MI*/1, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_TGTmul24_oneuse),

// GISEL: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// GISEL: GIM_CheckIntrinsicID, /*MI*/1, /*Op*/1, GIMT_Encode2(Intrinsic::tgt_mul24),
// GBUILTIN: GIM_CheckHasOneUse, /*MI*/1,
// GCUSTOM: GIM_CheckCxxInsnPredicate, /*MI*/1, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_TGTmul24_oneuse),

// GISEL: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(MyTarget::G_TGT_MUL24),
// GBUILTIN: GIM_CheckHasOneUse, /*MI*/1,
// GCUSTOM: GIM_CheckCxxInsnPredicate, /*MI*/1, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_TGTmul24_oneuse),

// GISEL: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(MyTarget::G_TGT_MUL24),
// GBUILTIN: GIM_CheckHasOneUse, /*MI*/1,
// GCUSTOM: GIM_CheckCxxInsnPredicate, /*MI*/1, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_TGTmul24_oneuse),
def inst_mad24 : I<
  (outs GPR32:$dst),
  (ins GPR32:$src0, GPR32:$src1, GPR32:$src2),
  [(set GPR32:$dst, (add (TGTmul24_oneuse i32:$src0, i32:$src1), i32:$src2))]>;
