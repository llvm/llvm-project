// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -optimize-match-table=false -I %p/../../../include -I %p/../Common %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def shiftl_1 : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() << 1, SDLoc(N), MVT::i32);
}]>;

def gi_shiftl_1 : GICustomOperandRenderer<"renderShiftImml1">,
  GISDNodeXFormEquiv<shiftl_1>;


def int_mytarget_sleep : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_mytarget_foo : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;


def SLEEP : I<(outs), (ins i32imm:$src0), []>;
def FOO : I<(outs GPR32:$dst), (ins GPR32:$src0, i32imm:$src1), []>;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC),
// GISEL: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, GIMT_Encode2(Intrinsic::mytarget_foo),
// GISEL: GIM_CheckIsImm, /*MI*/0, /*Op*/3,
// GISEL: GIR_CustomOperandRenderer, /*InsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, /*OperandRenderer*/GIMT_Encode2(GICR_renderShiftImml1), // src1
def : Pat<
  (int_mytarget_foo i32:$src0, (i32 timm:$src1)),
  (FOO GPR32:$src0, (shiftl_1 $src1))
>;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// GISEL: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, GIMT_Encode2(Intrinsic::mytarget_sleep),
// GISEL: GIM_CheckIsImm, /*MI*/0, /*Op*/1,
// GISEL: GIR_CustomOperandRenderer, /*InsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*OperandRenderer*/GIMT_Encode2(GICR_renderShiftImml1), // src0
def : Pat<
  (int_mytarget_sleep (i32 timm:$src0)),
  (SLEEP (shiftl_1 $src0))
>;
