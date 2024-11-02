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


// CHECK:      const uint8_t *MyTargetInstructionSelector::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(79), // Rule ID 3 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FMAXNUM),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] SelectSrcMods:src0:mods0
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/GIMT_Encode2(0), GIMT_Encode2(GICP_gi_SelectSrcMods),
// CHECK-NEXT:       // MIs[0] SelectSrcMods:src1:mods1
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/GIMT_Encode2(1), GIMT_Encode2(GICP_gi_SelectSrcMods),
// CHECK-NEXT:       // (fmaxnum:{ *:[f32] } (SelectSrcMods:{ *:[f32] } f32:{ *:[f32] }:$src0, src_mods:{ *:[i32] }:$mods0), (SelectSrcMods:{ *:[f32] } f32:{ *:[f32] }:$src1, src_mods:{ *:[i32] }:$mods1))  =>  (FMAX:{ *:[f32] } src_mods:{ *:[i32] }:$mods0, f32:{ *:[f32] }:$src0, src_mods:{ *:[i32] }:$mods1, f32:{ *:[f32] }:$src1)
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FMAX),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // mods0
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src0
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(1), /*SubOperand*/1, // mods1
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(1), /*SubOperand*/0, // src1
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 3,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 0: @79
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(139), // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FFLOOR),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] SelectClampOMod:src0:omod:clamp
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/GIMT_Encode2(0), GIMT_Encode2(GICP_gi_SelectClampOMod),
// CHECK-NEXT:       // (ffloor:{ *:[f32] } (SelectClampOMod:{ *:[f32] } f32:{ *:[f32] }:$src0, omod:{ *:[i32] }:$omod, i1:{ *:[i1] }:$clamp))  =>  (FLOMP:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp, omod:{ *:[i32] }:$omod)
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FLOMP),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src0
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/2, // clamp
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // omod
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 2,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 1: @139
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4(207), // Rule ID 8 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FCANONICALIZE),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] SelectSrcMods:src:mods
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/GIMT_Encode2(0), GIMT_Encode2(GICP_gi_SelectSrcMods),
// CHECK-NEXT:       // (fcanonicalize:{ *:[f32] } (SelectSrcMods:{ *:[f32] } f32:{ *:[f32] }:$src, i32:{ *:[i32] }:$mods))  =>  (FMAX:{ *:[f32] } ?:{ *:[i32] }:$mods, ?:{ *:[f32] }:$src, ?:{ *:[i32] }:$mods, ?:{ *:[f32] }:$src, 0:{ *:[i1] })
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FMAX),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // mods
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // mods
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 8,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 2: @207
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(265), // Rule ID 5 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FCOS),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] SelectOMod:src0:omod
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/GIMT_Encode2(0), GIMT_Encode2(GICP_gi_SelectOMod),
// CHECK-NEXT:       // (fcos:{ *:[f32] } (SelectOMod:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$omod))  =>  (FLAMP:{ *:[f32] } FPR32:{ *:[f32] }:$src0, omod:{ *:[i32] }:$omod)
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FLAMP),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src0
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // omod
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 5,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 3: @265
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(345), // Rule ID 7 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FEXP2),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] SelectClamp:src0:clamp
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/GIMT_Encode2(0), GIMT_Encode2(GICP_gi_SelectClamp),
// CHECK-NEXT:       // (fexp2:{ *:[f32] } (SelectClamp:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp))  =>  (FEEPLE:{ *:[f32] } FPR32:{ *:[f32] }:$src0, (FFOO:{ *:[f32] } FPR32:{ *:[f32] }:$src0), clamp:{ *:[i1] }:$clamp)
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(MyTarget::FFOO),
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/1, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src0
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/1, /*Imm*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FEEPLE),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src0
// CHECK-NEXT:       GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // clamp
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 7,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 4: @345
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(400), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FSIN),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] SelectClamp:src0:clamp
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/GIMT_Encode2(0), GIMT_Encode2(GICP_gi_SelectClamp),
// CHECK-NEXT:       // (fsin:{ *:[f32] } (SelectClamp:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp))  =>  (FFOO:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp)
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FFOO),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src0
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // clamp
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 5: @400
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(458), // Rule ID 6 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FSQRT),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] SelectClamp:src0:clamp
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckComplexPattern, /*MI*/0, /*Op*/1, /*Renderer*/GIMT_Encode2(0), GIMT_Encode2(GICP_gi_SelectClamp),
// CHECK-NEXT:       // (fsqrt:{ *:[f32] } (SelectClamp:{ *:[f32] } f32:{ *:[f32] }:$src0, i1:{ *:[i1] }:$clamp))  =>  (FLAMP:{ *:[f32] } FPR32:{ *:[f32] }:$src0, 93:{ *:[i32] }, clamp:{ *:[i1] }:$clamp)
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FLAMP),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, // src0
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/0, /*Imm*/93,
// CHECK-NEXT:       GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, // clamp
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 6,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 6: @458
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(503), // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_ROUND),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] src0
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       // (fround:{ *:[f32] } f32:{ *:[f32] }:$src0)  =>  (FBAR:{ *:[f32] } f32:{ *:[f32] }:$src0)
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FBAR),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 1,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 7: @503
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(548), // Rule ID 4 //
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_TRUNC),
// CHECK-NEXT:       // MIs[0] DstI[dst]
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR32RegClassID),
// CHECK-NEXT:       // MIs[0] src0
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       // (ftrunc:{ *:[f32] } f32:{ *:[f32] }:$src0)  =>  (FFOO:{ *:[f32] } FPR32:{ *:[f32] }:$src0)
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FFOO),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:       // GIR_Coverage, 4,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 8: @548
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: 549 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }

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
