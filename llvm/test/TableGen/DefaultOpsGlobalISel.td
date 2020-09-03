// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common -o - | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"


def SelectClamp  : ComplexPattern<untyped, 2, "SelectClamp">;
def SelectOMod  : ComplexPattern<untyped, 2, "SelectOMod">;
def SelectClampOMod  : ComplexPattern<untyped, 3, "SelectClampOMod">;
def SelectSrcMods  : ComplexPattern<untyped, 2, "SelectSrcMods">;

def gi_SelectClamp :
    GIComplexOperandMatcher<s32, "selectClamp">,
    GIComplexPatternEquiv<SelectClamp>;

def gi_SelectOMod :
    GIComplexOperandMatcher<s32, "selectOMod">,
    GIComplexPatternEquiv<SelectOMod>;

def gi_SelectClampOMod :
    GIComplexOperandMatcher<s32, "selectClampOMod">,
    GIComplexPatternEquiv<SelectClampOMod>;

def gi_SelectSrcMods :
    GIComplexOperandMatcher<s32, "selectSrcMods">,
    GIComplexPatternEquiv<SelectSrcMods>;


def src_mods : Operand <i32>;
def omod : OperandWithDefaultOps <i32, (ops (i32 0))>;
def clamp : OperandWithDefaultOps <i1, (ops (i1 0))>;


// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FMAXNUM,
// CHECK: GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/0, GICP_gi_SelectSrcMods,
// CHECK: GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/1, GICP_gi_SelectSrcMods,
// CHECK: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FMAX,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // mods0
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src0
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/1, /*SubOperand*/1, // mods1
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/1, /*SubOperand*/0, // src1
// CHECK-NEXT: GIR_AddImm, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,


// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FFLOOR,
// CHECK: GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/0, GICP_gi_SelectClampOMod,
// CHECK: // (ffloor:{ *:[f32] } (SelectClampOMod:{ *:[f32] } f32:{ *:[f32] }:$src0, omod:{ *:[i32] }:$omod, i1:{ *:[i1] }:$clamp))  =>  (FLOMP:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp, omod:{ *:[i32] }:$omod)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FLOMP,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src0
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/2, // clamp
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // omod


// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FCANONICALIZE,
// CHECK: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FMAX,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // mods
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // mods
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src
// CHECK-NEXT: GIR_AddImm, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,


// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FCOS,
// CHECK: // (fcos:{ *:[f32] } (SelectOMod:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$omod))  =>  (FLAMP:{ *:[f32] } FPR32:{ *:[f32] }:$src0, omod:{ *:[i32] }:$omod)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FLAMP,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src0
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // omod
// CHECK-NEXT: GIR_AddImm, /*InsnID*/0, /*Imm*/0,


// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FEXP2,
// CHECK: // (fexp2:{ *:[f32] } (SelectClamp:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp))  =>  (FEEPLE:{ *:[f32] } FPR32:{ *:[f32] }:$src0, (FFOO:{ *:[f32] } FPR32:{ *:[f32] }:$src0), clamp:{ *:[i1] }:$clamp)

// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/MyTarget::FFOO,
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/1, /*RendererID*/0, /*SubOperand*/0, // src0
// CHECK-NEXT: GIR_AddImm, /*InsnID*/1, /*Imm*/0,
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FEEPLE,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src0
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // clamp
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/0,


// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FSIN,
// CHECK: // (fsin:{ *:[f32] } (SelectClamp:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp))  =>  (FFOO:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FFOO,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src0
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // clamp
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,


// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FSQRT,
// CHECK: // (fsqrt:{ *:[f32] } (SelectClamp:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp))  =>  (FLAMP:{ *:[f32] } FPR32:{ *:[f32] }:$src0, 93:{ *:[i32] }, clamp:{ *:[i1] }:$clamp)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FLAMP,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src0
// CHECK-NEXT: GIR_AddImm, /*InsnID*/0, /*Imm*/93,
// CHECK-NEXT: GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // clamp
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,

// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_INTRINSIC_ROUND,
// CHECK: // (fround:{ *:[f32] } f32:{ *:[f32] }:$src0)  =>  (FBAR:{ *:[f32] } f32:{ *:[f32] }:$src0)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FBAR,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
// CHECK-NEXT: GIR_AddImm, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,

// CHECK: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_INTRINSIC_TRUNC,
// CHECK: // (ftrunc:{ *:[f32] } f32:{ *:[f32] }:$src0)  =>  (FFOO:{ *:[f32] } FPR32:{ *:[f32] }:$src0)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::FFOO,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
// CHECK-NEXT: GIR_AddImm, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,


// Have default operand with explicit value from complex pattern.
def FFOO : I<(outs FPR32:$dst), (ins FPR32:$src0, clamp:$clamp),
            [(set FPR32:$dst, (fsin (SelectClamp f32:$src0, i1:$clamp)))]>;


// Have default operand, not explicitly specified in a standalone
// pattern.
def : Pat <
  (ftrunc f32:$src0),
  (FFOO FPR32:$src0)
>;

// Have default operand, not explicitly specified in an instruction
// definition pattern.
def FBAR : I<(outs FPR32:$dst), (ins FPR32:$src0, clamp:$clamp),
            [(set FPR32:$dst, (fround f32:$src0))]>;


// // Swapped order in instruction from pattern
def FLOMP : I<
  (outs FPR32:$dst), (ins FPR32:$src0, clamp:$clamp, omod:$omod),
  [(set FPR32:$dst, (ffloor (SelectClampOMod f32:$src0, omod:$omod, i1:$clamp)))]>;

def FLAMP : I<(outs FPR32:$dst), (ins FPR32:$src0, omod:$omod, clamp:$clamp), []>;

// // Have 2 default operands, and the first is specified
def : Pat <
  (fcos (SelectOMod f32:$src0, i32:$omod)),
  (FLAMP FPR32:$src0, omod:$omod)
>;

// Immediate used for first defaulted operand
def : Pat <
  (fsqrt (SelectClamp f32:$src0, i1:$clamp)),
  (FLAMP FPR32:$src0, 93, clamp:$clamp)
>;

def FEEPLE : I<(outs FPR32:$dst),
               (ins FPR32:$src0, FPR32:$src1, clamp:$clamp), []>;

// Default operand isn't on the root ouput instruction
def : Pat <
  (fexp2 (SelectClamp f32:$src0, i1:$clamp)),
  (FEEPLE FPR32:$src0, (FFOO FPR32:$src0), clamp:$clamp)
>;

// Same instruction is used in two different pattern contexts, one
// uses the default and one does not.
def FMAX : I<(outs FPR32:$dst),
  (ins src_mods:$mods0, FPR32:$src0, src_mods:$mods1, FPR32:$src1, clamp:$clamp),
  [(set FPR32:$dst, (f32 (fmaxnum (SelectSrcMods f32:$src0, src_mods:$mods0),
                                  (SelectSrcMods f32:$src1, src_mods:$mods1))))]
>;

def : Pat<
  (fcanonicalize (f32 (SelectSrcMods f32:$src, i32:$mods))),
  (FMAX $mods, $src, $mods, $src, 0)
>;
