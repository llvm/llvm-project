// RUN: llvm-tblgen -gen-global-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

//===- Define the necessary boilerplate for our test target. --------------===//

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

let TargetPrefix = "mytarget" in {
def int_mytarget_nop : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
}

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
def GPR32Op : RegisterOperand<GPR32>;
def F0 : Register<"f0"> { let Namespace = "MyTarget"; }
def FPR32 : RegisterClass<"MyTarget", [f32], 32, (add F0)>;

class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

def complex : Operand<i32>, ComplexPattern<i32, 2, "SelectComplexPattern", []> {
  let MIOperandInfo = (ops i32imm, i32imm);
}
def gi_complex :
    GIComplexOperandMatcher<s32, "selectComplexPattern">,
    GIComplexPatternEquiv<complex>;
def complex_rr : Operand<i32>, ComplexPattern<i32, 2, "SelectComplexPatternRR", []> {
  let MIOperandInfo = (ops GPR32, GPR32);
}
def gi_complex_rr :
    GIComplexOperandMatcher<s32, "selectComplexPatternRR">,
    GIComplexPatternEquiv<complex_rr>;

def m1 : OperandWithDefaultOps <i32, (ops (i32 -1))>;
def Z : OperandWithDefaultOps <i32, (ops R0)>;
def m1Z : OperandWithDefaultOps <i32, (ops (i32 -1), R0)>;

def HasA : Predicate<"Subtarget->hasA()">;
def HasB : Predicate<"Subtarget->hasB()">;
def HasC : Predicate<"Subtarget->hasC()"> { let RecomputePerFunction = 1; }

//===- Test the function boilerplate. -------------------------------------===//

// CHECK: const unsigned MAX_SUBTARGET_PREDICATES = 3;
// CHECK: using PredicateBitset = llvm::PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_DECL
// CHECK-NEXT:    mutable MatcherState State;
// CHECK-NEXT:    typedef ComplexRendererFns(MyTargetInstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;
// CHECK-NEXT:    const MatcherInfoTy<PredicateBitset, ComplexMatcherMemFn> MatcherInfo;
// CHECK-NEXT:    static MyTargetInstructionSelector::ComplexMatcherMemFn ComplexPredicateFns[];
// CHECK-NEXT:  #endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_INIT
// CHECK-NEXT:    , State(2),
// CHECK-NEXT:    MatcherInfo({TypeObjects, FeatureBitsets, I64ImmPredicateFns, APIntImmPredicateFns, APFloatImmPredicateFns, ComplexPredicateFns})
// CHECK-NEXT:  #endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT

// CHECK-LABEL: enum SubtargetFeatureBits : uint8_t {
// CHECK-NEXT:    Feature_HasABit = 0,
// CHECK-NEXT:    Feature_HasBBit = 1,
// CHECK-NEXT:    Feature_HasCBit = 2,
// CHECK-NEXT:  };

// CHECK-LABEL: PredicateBitset MyTargetInstructionSelector::
// CHECK-NEXT:  computeAvailableModuleFeatures(const MyTargetSubtarget *Subtarget) const {
// CHECK-NEXT:    PredicateBitset Features;
// CHECK-NEXT:    if (Subtarget->hasA())
// CHECK-NEXT:      Features[Feature_HasABit] = 1;
// CHECK-NEXT:    if (Subtarget->hasB())
// CHECK-NEXT:      Features[Feature_HasBBit] = 1;
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: PredicateBitset MyTargetInstructionSelector::
// CHECK-NEXT:  computeAvailableFunctionFeatures(const MyTargetSubtarget *Subtarget, const MachineFunction *MF) const {
// CHECK-NEXT:    PredicateBitset Features;
// CHECK-NEXT:    if (Subtarget->hasC())
// CHECK-NEXT:      Features[Feature_HasCBit] = 1;
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: // LLT Objects.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GILLT_s32,
// CHECK-NEXT:  }
// CHECK-NEXT:  const static LLT TypeObjects[] = {
// CHECK-NEXT:    LLT::scalar(32),
// CHECK-NEXT:  };

// CHECK-LABEL: // Feature bitsets.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIFBS_Invalid,
// CHECK-NEXT:    GIFBS_HasA,
// CHECK-NEXT:    GIFBS_HasA_HasB_HasC,
// CHECK-NEXT:  }
// CHECK-NEXT:  const static PredicateBitset FeatureBitsets[] {
// CHECK-NEXT:    {}, // GIFBS_Invalid
// CHECK-NEXT:    {Feature_HasABit, },
// CHECK-NEXT:    {Feature_HasABit, Feature_HasBBit, Feature_HasCBit, },
// CHECK-NEXT:  };

// CHECK-LABEL: // ComplexPattern predicates.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GICP_Invalid,
// CHECK-NEXT:    GICP_gi_complex,
// CHECK-NEXT:    GICP_gi_complex_rr,
// CHECK-NEXT:  };

// CHECK-LABEL: // PatFrag predicates.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIPFP_I64_Predicate_simm8 = GIPFP_I64_Invalid + 1,
// CHECK-NEXT:  };
// CHECK-NEXT:  static bool Predicate_simm8(int64_t Imm) { return isInt<8>(Imm);   }
// CHECK-NEXT:  static InstructionSelector::I64ImmediatePredicateFn I64ImmPredicateFns[] = {
// CHECK-NEXT:    nullptr,
// CHECK-NEXT:    Predicate_simm8,
// CHECK-NEXT:  };

// CHECK-LABEL: // PatFrag predicates.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIPFP_APFloat_Predicate_fpimmz = GIPFP_APFloat_Invalid + 1,
// CHECK-NEXT:  };
// CHECK-NEXT:  static bool Predicate_fpimmz(const APFloat & Imm) { return Imm->isExactlyValue(0.0); }
// CHECK-NEXT:  static InstructionSelector::APFloatImmediatePredicateFn APFloatImmPredicateFns[] = {
// CHECK-NEXT:    nullptr,
// CHECK-NEXT:    Predicate_fpimmz,
// CHECK-NEXT:  };

// CHECK-LABEL: // PatFrag predicates.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIPFP_APInt_Predicate_simm9 = GIPFP_APInt_Invalid + 1,
// CHECK-NEXT:  };
// CHECK-NEXT:  static bool Predicate_simm9(const APInt & Imm) { return isInt<9>(Imm->getSExtValue());   }
// CHECK-NEXT:  static InstructionSelector::APIntImmediatePredicateFn APIntImmPredicateFns[] = {
// CHECK-NEXT:    nullptr,
// CHECK-NEXT:    Predicate_simm9,
// CHECK-NEXT:  };

// CHECK-LABEL: MyTargetInstructionSelector::ComplexMatcherMemFn
// CHECK-NEXT:  MyTargetInstructionSelector::ComplexPredicateFns[] = {
// CHECK-NEXT:    nullptr, // GICP_Invalid
// CHECK-NEXT:    &MyTargetInstructionSelector::selectComplexPattern, // gi_complex
// CHECK-NEXT:    &MyTargetInstructionSelector::selectComplexPatternRR, // gi_complex_rr
// CHECK-NEXT:  }

// CHECK: bool MyTargetInstructionSelector::selectImpl(MachineInstr &I) const {
// CHECK-NEXT: MachineFunction &MF = *I.getParent()->getParent();
// CHECK-NEXT: MachineRegisterInfo &MRI = MF.getRegInfo();
// CHECK:      AvailableFunctionFeatures = computeAvailableFunctionFeatures(&STI, &MF);
// CHECK-NEXT: const PredicateBitset AvailableFeatures = getAvailableFeatures();
// CHECK-NEXT: NewMIVector OutMIs;
// CHECK-NEXT: State.MIs.clear();
// CHECK-NEXT: State.MIs.push_back(&I);

//===- Test a pattern with multiple ComplexPatterns in multiple instrs ----===//
//

// CHECK-LABEL: MatchTable0[] = {
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 0*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SELECT,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex,
// CHECK-NEXT:    // MIs[0] src3
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/3, /*Renderer*/1, GICP_gi_complex,
// CHECK-NEXT:    // (select:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src2, complex:{ *:[i32] }:$src3) => (INSN2:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src3, complex:{ *:[i32] }:$src2)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN2,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/1,
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/0,
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 0: @[[LABEL]]

def INSN3 : I<(outs GPR32:$dst),
              (ins GPR32Op:$src1, GPR32:$src2a, GPR32:$src2b, GPR32:$src3, complex:$src4, i32imm:$src5a, i32imm:$src5b), []>;
def : Pat<(select GPR32:$src1, (complex_rr GPR32:$src2a, GPR32:$src2b),
                               (select GPR32:$src3,
                                       complex:$src4,
                                       (complex i32imm:$src5a, i32imm:$src5b))),
          (INSN3 GPR32:$src1, GPR32:$src2b, GPR32:$src2a, GPR32:$src3,
                 complex:$src4, i32imm:$src5a, i32imm:$src5b)>;

//===- Test a pattern with multiple ComplexPattern operands. --------------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 1*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// CHECK-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/4,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SELECT,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex_rr,
// CHECK-NEXT:    // MIs[0] Operand 3
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SELECT,
// CHECK-NEXT:    // MIs[1] Operand 0
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    // MIs[1] src3
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[1] src4
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/1, /*Op*/2, /*Renderer*/1, GICP_gi_complex,
// CHECK-NEXT:    // MIs[1] Operand 3
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/1, /*Op*/3, /*Renderer*/2, GICP_gi_complex,
// CHECK-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:    // (select:{ *:[i32] } GPR32:{ *:[i32] }:$src1, (complex_rr:{ *:[i32] } GPR32:{ *:[i32] }:$src2a, GPR32:{ *:[i32] }:$src2b), (select:{ *:[i32] } GPR32:{ *:[i32] }:$src3, complex:{ *:[i32] }:$src4, (complex:{ *:[i32] } i32imm:{ *:[i32] }:$src5a, i32imm:{ *:[i32] }:$src5b)))  =>  (INSN3:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2b, GPR32:{ *:[i32] }:$src2a, GPR32:{ *:[i32] }:$src3, complex:{ *:[i32] }:$src4, i32imm:{ *:[i32] }:$src5a, i32imm:{ *:[i32] }:$src5b)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN3,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // src2b
// CHECK-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src2a
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src3
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/1,
// CHECK-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/2, /*SubOperand*/0, // src5a
// CHECK-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/2, /*SubOperand*/1, // src5b
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 1: @[[LABEL]]

def : GINodeEquiv<G_SELECT, select>;
let mayLoad = 1 in {
  def INSN2 : I<(outs GPR32:$dst), (ins GPR32Op:$src1, complex:$src2, complex:$src3), []>;
}
def : Pat<(select GPR32:$src1, complex:$src2, complex:$src3),
          (INSN2 GPR32:$src1, complex:$src3, complex:$src2)>;

//===- Test a simple pattern with regclass operands. ----------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 2*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ADD,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID
// CHECK-NEXT:    // MIs[0] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // (add:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2) => (ADD:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::ADD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 2: @[[LABEL]]

def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2),
            [(set GPR32:$dst, (add GPR32:$src1, GPR32:$src2))]>;

//===- Test a pattern with a tied operand in the matcher ------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 3*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ADD,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src{{$}}
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src{{$}}
// CHECK-NEXT:    GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
// CHECK-NEXT:    // (add:{ *:[i32] } GPR32:{ *:[i32] }:$src, GPR32:{ *:[i32] }:$src) => (DOUBLE:{ *:[i32] } GPR32:{ *:[i32] }:$src)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::DOUBLE,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 3: @[[LABEL]]

def DOUBLE : I<(outs GPR32:$dst), (ins GPR32:$src), [(set GPR32:$dst, (add GPR32:$src, GPR32:$src))]>;

//===- Test a simple pattern with ValueType operands. ----------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 4*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ADD,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    // MIs[0] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    // (add:{ *:[i32] } i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) => (ADD:{ *:[i32] } i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::ADD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 4: @[[LABEL]]

def : Pat<(add i32:$src1, i32:$src2),
          (ADD i32:$src1, i32:$src2)>;

//===- Test a simple pattern with an intrinsic. ---------------------------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 5*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_INTRINSIC,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::mytarget_nop,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // (intrinsic_wo_chain:{ *:[i32] } [[ID:[0-9]+]]:{ *:[iPTR] }, GPR32:{ *:[i32] }:$src1) => (MOV:{ *:[i32] } GPR32:{ *:[i32] }:$src1)

// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOV,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 5: @[[LABEL]]

def MOV : I<(outs GPR32:$dst), (ins GPR32:$src1),
            [(set GPR32:$dst, (int_mytarget_nop GPR32:$src1))]>;

//===- Test a nested instruction match. -----------------------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 6*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HasA,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
// CHECK-NEXT:    // MIs[1] Operand 0
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    // MIs[1] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[1] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src3
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:    // (mul:{ *:[i32] } (add:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2), GPR32:{ *:[i32] }:$src3)  =>  (MULADD:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src3)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MULADD,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 6: @[[LABEL]]

// We also get a second rule by commutativity.
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 7*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HasA,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src3
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
// CHECK-NEXT:    // MIs[1] Operand 0
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    // MIs[1] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[1] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:    // (mul:{ *:[i32] } GPR32:{ *:[i32] }:$src3, (add:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2))  =>  (MULADD:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src3)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MULADD,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 7: @[[LABEL]]

def MULADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3),
               [(set GPR32:$dst,
                     (mul (add GPR32:$src1, GPR32:$src2), GPR32:$src3))]>,
             Requires<[HasA]>;

//===- Test another simple pattern with regclass operands. ----------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 8*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HasA_HasB_HasC,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // (mul:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2) => (MUL:{ *:[i32] } GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MUL,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 8: @[[LABEL]]

def MUL : I<(outs GPR32:$dst), (ins GPR32:$src2, GPR32:$src1),
             [(set GPR32:$dst, (mul GPR32:$src1, GPR32:$src2))]>,
          Requires<[HasA, HasB, HasC]>;

//===- Test a more complex multi-instruction match. -----------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 9*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HasA,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// CHECK-NEXT:    GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/2, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SUB,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
// CHECK-NEXT:    // MIs[1] Operand 0
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    // MIs[1] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[1] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_SUB,
// CHECK-NEXT:    // MIs[2] Operand 0
// CHECK-NEXT:    GIM_CheckType, /*MI*/2, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    // MIs[2] src3
// CHECK-NEXT:    GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[2] src4
// CHECK-NEXT:    GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/2, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:    // (sub:{ *:[i32] } (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2), (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src3, GPR32:{ *:[i32] }:$src4)) => (INSNBOB:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src3, GPR32:{ *:[i32] }:$src4)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSNBOB,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src3
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // src4
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 9: @[[LABEL]]

def INSNBOB : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3, GPR32:$src4),
                 [(set GPR32:$dst,
                      (sub (sub GPR32:$src1, GPR32:$src2), (sub GPR32:$src3, GPR32:$src4)))]>,
               Requires<[HasA]>;

//===- Test a pattern with ComplexPattern operands. -----------------------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 10*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SUB,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex,
// CHECK-NEXT:    // (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src2) => (INSN1:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src2)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN1,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 10: @[[LABEL]]

def INSN1 : I<(outs GPR32:$dst), (ins GPR32:$src1, complex:$src2), []>;
def : Pat<(sub GPR32:$src1, complex:$src2), (INSN1 GPR32:$src1, complex:$src2)>;

//===- Test a simple pattern with a default operand. ----------------------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 11*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -2
// CHECK-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -2:{ *:[i32] }) => (XORI:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORI,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 11: @[[LABEL]]

// The -2 is just to distinguish it from the 'not' case below.
def XORI : I<(outs GPR32:$dst), (ins m1:$src2, GPR32:$src1),
             [(set GPR32:$dst, (xor GPR32:$src1, -2))]>;

//===- Test a simple pattern with a default register operand. -------------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 12*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -3
// CHECK-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -3:{ *:[i32] }) => (XOR:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XOR,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 12: @[[LABEL]]

// The -3 is just to distinguish it from the 'not' case below and the other default op case above.
def XOR : I<(outs GPR32:$dst), (ins Z:$src2, GPR32:$src1),
            [(set GPR32:$dst, (xor GPR32:$src1, -3))]>;

//===- Test a simple pattern with a multiple default operands. ------------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 13*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -4
// CHECK-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -4:{ *:[i32] }) => (XORlike:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORlike,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 13: @[[LABEL]]

// The -4 is just to distinguish it from the other 'not' cases.
def XORlike : I<(outs GPR32:$dst), (ins m1Z:$src2, GPR32:$src1),
                [(set GPR32:$dst, (xor GPR32:$src1, -4))]>;

//===- Test a simple pattern with multiple operands with defaults. --------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 14*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -5,
// CHECK-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -5:{ *:[i32] }) => (XORManyDefaults:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORManyDefaults,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 14: @[[LABEL]]

// The -5 is just to distinguish it from the other cases.
def XORManyDefaults : I<(outs GPR32:$dst), (ins m1Z:$src3, Z:$src2, GPR32:$src1),
                        [(set GPR32:$dst, (xor GPR32:$src1, -5))]>;

//===- Test a simple pattern with constant immediate operands. ------------===//
//
// This must precede the 3-register variants because constant immediates have
// priority over register banks.

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 15*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Wm
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
// CHECK-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$Wm, -1:{ *:[i32] }) => (ORN:{ *:[i32] } R0:{ *:[i32] }, GPR32:{ *:[i32] }:$Wm)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::ORN,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // Wm
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 15: @[[LABEL]]

def ORN : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def : Pat<(not GPR32:$Wm), (ORN R0, GPR32:$Wm)>;

//===- Test a COPY_TO_REGCLASS --------------------------------------------===//
//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 16*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_BITCAST,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::FPR32RegClassID,
// CHECK-NEXT:    // (bitconvert:{ *:[i32] } FPR32:{ *:[f32] }:$src1) => (COPY_TO_REGCLASS:{ *:[i32] } FPR32:{ *:[f32] }:$src1, GPR32:{ *:[i32] })
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
// CHECK-NEXT:    GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GPR32*/1,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 16: @[[LABEL]]

def : Pat<(i32 (bitconvert FPR32:$src1)),
          (COPY_TO_REGCLASS FPR32:$src1, GPR32)>;

//===- Test a simple pattern with just a specific leaf immediate. ---------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 17*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 1,
// CHECK-NEXT:    // 1:{ *:[i32] }  =>  (MOV1:{ *:[i32] })
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOV1,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 17: @[[LABEL]]

def MOV1 : I<(outs GPR32:$dst), (ins), [(set GPR32:$dst, 1)]>;

//===- Test a simple pattern with a leaf immediate and a predicate. -------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 18*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// CHECK-NEXT:    GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_simm8,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    // No operand predicates
// CHECK-NEXT:    // (imm:{ *:[i32] })<<P:Predicate_simm8>>:$imm => (MOVimm8:{ *:[i32] } (imm:{ *:[i32] }):$imm)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVimm8,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 18: @[[LABEL]]

def simm8 : ImmLeaf<i32, [{ return isInt<8>(Imm); }]>;
def MOVimm8 : I<(outs GPR32:$dst), (ins i32imm:$imm), [(set GPR32:$dst, simm8:$imm)]>;

//===- Same again but use an IntImmLeaf. ----------------------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 19*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// CHECK-NEXT:    GIM_CheckAPIntImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APInt_Predicate_simm9,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    // No operand predicates
// CHECK-NEXT:    // (imm:{ *:[i32] })<<P:Predicate_simm9>>:$imm =>  (MOVimm9:{ *:[i32] } (imm:{ *:[i32] }):$imm)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVimm9,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 19: @[[LABEL]]

def simm9 : IntImmLeaf<i32, [{ return isInt<9>(Imm->getSExtValue()); }]>;
def MOVimm9 : I<(outs GPR32:$dst), (ins i32imm:$imm), [(set GPR32:$dst, simm9:$imm)]>;

//===- Test a simple pattern with just a leaf immediate. ------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 20*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    // No operand predicates
// CHECK-NEXT:    // (imm:{ *:[i32] }):$imm =>  (MOVimm:{ *:[i32] } (imm:{ *:[i32] }):$imm)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVimm,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 20: @[[LABEL]]

def MOVimm : I<(outs GPR32:$dst), (ins i32imm:$imm), [(set GPR32:$dst, imm:$imm)]>;

//===- Test a simple pattern with a FP immediate and a predicate. ---------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 21*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FCONSTANT,
// CHECK-NEXT:    GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimmz,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::FPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    // No operand predicates
// CHECK-NEXT:    // (fpimm:{ *:[f32] })<<P:Predicate_fpimmz>>:$imm =>  (MOVfpimmz:{ *:[f32] } (fpimm:{ *:[f32] }):$imm)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVfpimmz,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_CopyFConstantAsFPImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 21: @[[LABEL]]

def fpimmz : FPImmLeaf<f32, [{ return Imm->isExactlyValue(0.0); }]>;
def MOVfpimmz : I<(outs FPR32:$dst), (ins f32imm:$imm), [(set FPR32:$dst, fpimmz:$imm)]>;

//===- Test a simple pattern with inferred pointer operands. ---------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 22*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// CHECK-NEXT:    GIM_CheckNonAtomic, /*MI*/0,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // (ld:{ *:[i32] } GPR32:{ *:[i32] }:$src1)<<P:Predicate_unindexedload>><<P:Predicate_load>> => (LOAD:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::LOAD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 22: @[[LABEL]]

def LOAD : I<(outs GPR32:$dst), (ins GPR32:$src1),
            [(set GPR32:$dst, (load GPR32:$src1))]>;
//===- Test a pattern with an MBB operand. --------------------------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 23*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/1,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_BR,
// CHECK-NEXT:    // MIs[0] target
// CHECK-NEXT:    GIM_CheckIsMBB, /*MI*/0, /*Op*/0,
// CHECK-NEXT:    // (br (bb:{ *:[Other] }):$target) => (BR (bb:{ *:[Other] }):$target)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::BR,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 23: @[[LABEL]]

def BR : I<(outs), (ins unknown:$target),
            [(br bb:$target)]>;

// CHECK-NEXT:    GIM_Reject,
// CHECK-NEXT:  };
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable0, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }
