// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common -optimize-match-table=false %s -o %T/non-optimized.cpp
// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common -optimize-match-table=true  %s -o %T/optimized.cpp
// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s -o %T/default.cpp

// RUN: FileCheck %s --check-prefixes=CHECK,R19C,R19N -input-file=%T/non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK,R19C,R19O -input-file=%T/optimized.cpp

// RUN: FileCheck %s --check-prefixes=CHECK,R21C,R21N -input-file=%T/non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK,R21C,R21O -input-file=%T/optimized.cpp

// RUN: FileCheck %s --check-prefixes=CHECK,R20C,R20N -input-file=%T/non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK,R20C,R20O -input-file=%T/optimized.cpp

// RUN: FileCheck %s --check-prefixes=CHECK,R00C,R00N -input-file=%T/non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK,R00C,R00O -input-file=%T/optimized.cpp

// RUN: FileCheck %s --check-prefixes=CHECK,R01C,R01N -input-file=%T/non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK,R01C,R01O -input-file=%T/optimized.cpp

// RUN: FileCheck %s --check-prefixes=CHECK,R02C,R02N,NOOPT -input-file=%T/non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK,R02C,R02O       -input-file=%T/optimized.cpp

// RUN: diff %T/default.cpp %T/optimized.cpp

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

//===- Define the necessary boilerplate for our test target. --------------===//

let TargetPrefix = "mytarget" in {
def int_mytarget_nop : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
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

def cimm8_xform : SDNodeXForm<imm, [{
    uint64_t Val = N->getZExtValue() << 1;
    return CurDAG->getTargetConstant(Val, SDLoc(N), MVT::i64);
  }]>;

def cimm8 : Operand<i32>, ImmLeaf<i32, [{return isInt<8>(Imm);}], cimm8_xform>;

def gi_cimm8 : GICustomOperandRenderer<"renderImm8">,
                GISDNodeXFormEquiv<cimm8_xform>;

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
// CHECK-NEXT:    typedef void(MyTargetInstructionSelector::*CustomRendererFn)(MachineInstrBuilder &, const MachineInstr&, int) const;
// CHECK-NEXT:    const ISelInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn> ISelInfo;
// CHECK-NEXT:    static MyTargetInstructionSelector::ComplexMatcherMemFn ComplexPredicateFns[];
// CHECK-NEXT:    static MyTargetInstructionSelector::CustomRendererFn CustomRenderers[];
// CHECK-NEXT:    bool testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const override;
// CHECK-NEXT:    bool testImmPredicate_APInt(unsigned PredicateID, const APInt &Imm) const override;
// CHECK-NEXT:    bool testImmPredicate_APFloat(unsigned PredicateID, const APFloat &Imm) const override;
// CHECK-NEXT:    const int64_t *getMatchTable() const override;
// CHECK-NEXT:    bool testMIPredicate_MI(unsigned PredicateID, const MachineInstr &MI) const override;
// CHECK-NEXT:  #endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_INIT
// CHECK-NEXT:    , State(2),
// CHECK-NEXT:    ISelInfo(TypeObjects, NumTypeObjects, FeatureBitsets, ComplexPredicateFns, CustomRenderers)
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
// CHECK-NEXT:      Features.set(Feature_HasABit);
// CHECK-NEXT:    if (Subtarget->hasB())
// CHECK-NEXT:      Features.set(Feature_HasBBit);
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: PredicateBitset MyTargetInstructionSelector::
// CHECK-NEXT:  computeAvailableFunctionFeatures(const MyTargetSubtarget *Subtarget, const MachineFunction *MF) const {
// CHECK-NEXT:    PredicateBitset Features;
// CHECK-NEXT:    if (Subtarget->hasC())
// CHECK-NEXT:      Features.set(Feature_HasCBit);
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: // LLT Objects.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GILLT_p0s32
// CHECK-NEXT:    GILLT_s32,
// CHECK-NEXT:  }
// CHECK-NEXT:  const static size_t NumTypeObjects = 2;
// CHECK-NEXT:  const static LLT TypeObjects[] = {
// CHECK-NEXT:    LLT::pointer(0, 32),
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
// CHECK-NEXT:    GIPFP_I64_Predicate_cimm8 = GIPFP_I64_Invalid + 1,
// CHECK-NEXT:    GIPFP_I64_Predicate_simm8,
// CHECK-NEXT:  };


// CHECK-NEXT: bool MyTargetInstructionSelector::testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const {
// CHECK-NEXT:   switch (PredicateID) {
// CHECK-NEXT:   case GIPFP_I64_Predicate_cimm8: {
// CHECK-NEXT:     return isInt<8>(Imm);
// CHECK-NEXT:     llvm_unreachable("ImmediateCode should have returned");
// CHECK-NEXT:     return false;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GIPFP_I64_Predicate_simm8: {
// CHECK-NEXT:      return isInt<8>(Imm);
// CHECK-NEXT:     llvm_unreachable("ImmediateCode should have returned");
// CHECK-NEXT:     return false;
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown predicate");
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// CHECK-LABEL: // PatFrag predicates.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIPFP_APFloat_Predicate_fpimmz = GIPFP_APFloat_Invalid + 1,
// CHECK-NEXT:  };
// CHECK-NEXT:  bool MyTargetInstructionSelector::testImmPredicate_APFloat(unsigned PredicateID, const APFloat & Imm) const {
// CHECK-NEXT:    switch (PredicateID) {
// CHECK-NEXT:    case GIPFP_APFloat_Predicate_fpimmz: {
// CHECK-NEXT:      return Imm->isExactlyValue(0.0);
// CHECK-NEXT:      llvm_unreachable("ImmediateCode should have returned");
// CHECK-NEXT:      return false;
// CHECK-NEXT:    }
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm_unreachable("Unknown predicate");
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }

// CHECK-LABEL: // PatFrag predicates.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIPFP_APInt_Predicate_simm9 = GIPFP_APInt_Invalid + 1,
// CHECK-NEXT:  };
// CHECK-NEXT:  bool MyTargetInstructionSelector::testImmPredicate_APInt(unsigned PredicateID, const APInt & Imm) const {
// CHECK-NEXT:    switch (PredicateID) {
// CHECK-NEXT:    case GIPFP_APInt_Predicate_simm9: {
// CHECK-NEXT:      return isInt<9>(Imm->getSExtValue());
// CHECK-NEXT:      llvm_unreachable("ImmediateCode should have returned");
// CHECK-NEXT:      return false;
// CHECK-NEXT:    }
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm_unreachable("Unknown predicate");
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }

// CHECK-LABEL: MyTargetInstructionSelector::ComplexMatcherMemFn
// CHECK-NEXT:  MyTargetInstructionSelector::ComplexPredicateFns[] = {
// CHECK-NEXT:    nullptr, // GICP_Invalid
// CHECK-NEXT:    &MyTargetInstructionSelector::selectComplexPattern, // gi_complex
// CHECK-NEXT:    &MyTargetInstructionSelector::selectComplexPatternRR, // gi_complex_rr
// CHECK-NEXT:  }

// CHECK-LABEL: // Custom renderers.
// CHECK-NEXT: enum {
// CHECK-NEXT:   GICR_Invalid,
// CHECK-NEXT:   GICR_renderImm8,
// CHECK-NEXT: };
// CHECK-NEXT: MyTargetInstructionSelector::CustomRendererFn
// CHECK-NEXT: MyTargetInstructionSelector::CustomRenderers[] = {
// CHECK-NEXT:   nullptr, // GICR_Invalid
// CHECK-NEXT:   &MyTargetInstructionSelector::renderImm8, // gi_cimm8
// CHECK-NEXT: };

// CHECK: bool MyTargetInstructionSelector::selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const {
// CHECK-NEXT: MachineFunction &MF = *I.getParent()->getParent();
// CHECK-NEXT: MachineRegisterInfo &MRI = MF.getRegInfo();
// CHECK-NEXT: const PredicateBitset AvailableFeatures = getAvailableFeatures();
// CHECK-NEXT: NewMIVector OutMIs;
// CHECK-NEXT: State.MIs.clear();
// CHECK-NEXT: State.MIs.push_back(&I);

// CHECK:      if (executeMatchTable(*this, OutMIs, State, ISelInfo, getMatchTable(), TII, MRI, TRI, RBI, AvailableFeatures, CoverageInfo)) {
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

// CHECK: const int64_t *
// CHECK-LABEL: MyTargetInstructionSelector::getMatchTable() const {
// CHECK-NEXT: MatchTable0[] = {

//===- Test a pattern with multiple ComplexPatterns in multiple instrs ----===//
//
// R19O-NEXT:  GIM_SwitchOpcode, /*MI*/0, /*[*/{{[0-9]+}}, {{[0-9]+}}, /*)*//*default:*//*Label [[DEFAULT_NUM:[0-9]+]]*/ [[DEFAULT:[0-9]+]],
// R19O-NEXT:  /*TargetOpcode::G_ADD*//*Label [[CASE_ADD_NUM:[0-9]+]]*/ [[CASE_ADD:[0-9]+]],
// R19O:       /*TargetOpcode::G_SELECT*//*Label [[CASE_SELECT_NUM:[0-9]+]]*/ [[CASE_SELECT:[0-9]+]],
// R19O:       // Label [[CASE_ADD_NUM]]: @[[CASE_ADD]]
// R19O:       // Label [[CASE_SELECT_NUM]]: @[[CASE_SELECT]]
// R19O-NEXT:  GIM_Try, /*On fail goto*//*Label [[GROUP_NUM:[0-9]+]]*/ [[GROUP:[0-9]+]],
// R19O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R19O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R19O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R19O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
//
// R19C-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
//
// R19O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R19O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R19N-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// R19N-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SELECT,
// R19N-NEXT:    // MIs[0] dst
// R19N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R19N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R19N-NEXT:    // MIs[0] src1
// R19N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R19N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R19N-NEXT:    // MIs[0] Operand 2
// R19N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
//
// R19N-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex_rr,
// R19N-NEXT:    // MIs[0] Operand 3
// R19N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
// R19C-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
// R19N-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/4,
// R19C-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SELECT,
// R19N-NEXT:    // MIs[1] Operand 0
// R19N-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// R19N-NEXT:    // MIs[1] src3
// R19C-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// R19O-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// R19O-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/3, /*Type*/GILLT_s32,
// R19N-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R19N-NEXT:    // MIs[1] src4
// R19N-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// R19N-NEXT:    GIM_CheckComplexPattern, /*MI*/1, /*Op*/2, /*Renderer*/1, GICP_gi_complex,
// R19N-NEXT:    // MIs[1] Operand 3
// R19N-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/3, /*Type*/GILLT_s32,
// R19N-NEXT:    GIM_CheckComplexPattern, /*MI*/1, /*Op*/3, /*Renderer*/2, GICP_gi_complex,
// R19O-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R19C-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// R19O-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex_rr,
// R19O-NEXT:    GIM_CheckComplexPattern, /*MI*/1, /*Op*/2, /*Renderer*/1, GICP_gi_complex,
// R19O-NEXT:    GIM_CheckComplexPattern, /*MI*/1, /*Op*/3, /*Renderer*/2, GICP_gi_complex,
// R19C-NEXT:    // (select:{ *:[i32] } GPR32:{ *:[i32] }:$src1, (complex_rr:{ *:[i32] } GPR32:{ *:[i32] }:$src2a, GPR32:{ *:[i32] }:$src2b), (select:{ *:[i32] } GPR32:{ *:[i32] }:$src3, complex:{ *:[i32] }:$src4, (complex:{ *:[i32] } i32imm:{ *:[i32] }:$src5a, i32imm:{ *:[i32] }:$src5b)))  =>  (INSN3:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2b, GPR32:{ *:[i32] }:$src2a, (INSN4:{ *:[i32] } GPR32:{ *:[i32] }:$src3, complex:{ *:[i32] }:$src4, i32imm:{ *:[i32] }:$src5a, i32imm:{ *:[i32] }:$src5b))
// R19C-NEXT:    GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// R19C-NEXT:    GIR_BuildMI, /*InsnID*/1, /*Opcode*/MyTarget::INSN4,
// R19C-NEXT:    GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
// R19C-NEXT:    GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/1, // src3
// R19C-NEXT:    GIR_ComplexRenderer, /*InsnID*/1, /*RendererID*/1,
// R19C-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/1, /*RendererID*/2, /*SubOperand*/0, // src5a
// R19C-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/1, /*RendererID*/2, /*SubOperand*/1, // src5b
// R19C-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// R19C-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN3,
// R19C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// R19C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// R19C-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/1, // src2b
// R19C-NEXT:    GIR_ComplexSubOperandRenderer, /*InsnID*/0, /*RendererID*/0, /*SubOperand*/0, // src2a
// R19C-NEXT:    GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// R19C-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// R19C-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// R19C-NEXT:    // GIR_Coverage, 19,
// R19C-NEXT:    GIR_Done,
// R19C-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
//
// R19O:       // Label [[GROUP_NUM]]: @[[GROUP]]
// R19O-NEXT:  GIM_Reject,
// R19O:       // Label [[DEFAULT_NUM]]: @[[DEFAULT]]
// R19O-NEXT:  GIM_Reject,
// R19O-NEXT:  };

def INSN3 : I<(outs GPR32:$dst),
              (ins GPR32Op:$src1, GPR32:$src2a, GPR32:$src2b, GPR32:$scr), []>;
def INSN4 : I<(outs GPR32:$scr),
              (ins GPR32:$src3, complex:$src4, i32imm:$src5a, i32imm:$src5b), []>;
def : Pat<(select GPR32:$src1, (complex_rr GPR32:$src2a, GPR32:$src2b),
                               (select GPR32:$src3,
                                       complex:$src4,
                                       (complex i32imm:$src5a, i32imm:$src5b))),
          (INSN3 GPR32:$src1, GPR32:$src2b, GPR32:$src2a,
                 (INSN4 GPR32:$src3, complex:$src4, i32imm:$src5a,
                        i32imm:$src5b))>;

// R21O-NEXT:  GIM_SwitchOpcode, /*MI*/0, /*[*/{{[0-9]+}}, {{[0-9]+}}, /*)*//*default:*//*Label [[DEFAULT_NUM:[0-9]+]]*/ [[DEFAULT:[0-9]+]],
// R21O-NEXT:  /*TargetOpcode::G_ADD*//*Label [[CASE_ADD_NUM:[0-9]+]]*/ [[CASE_ADD:[0-9]+]],
// R21O:       /*TargetOpcode::G_SELECT*//*Label [[CASE_SELECT_NUM:[0-9]+]]*/ [[CASE_SELECT:[0-9]+]],
// R21O:       // Label [[CASE_ADD_NUM]]: @[[CASE_ADD]]
// R21O:       // Label [[CASE_SELECT_NUM]]: @[[CASE_SELECT]]
// R21O-NEXT:  GIM_Try, /*On fail goto*//*Label [[GROUP_NUM:[0-9]+]]*/ [[GROUP:[0-9]+]],
// R21O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R21O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R21O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R21O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
//
// R21C-NEXT:  GIM_Try, /*On fail goto*//*Label [[PREV_NUM:[0-9]+]]*/ [[PREV:[0-9]+]], // Rule ID 19 //
// R21C-NOT:     GIR_Done,
// R21C:         // GIR_Coverage, 19,
// R21C-NEXT:    GIR_Done,
// R21C-NEXT:  // Label [[PREV_NUM]]: @[[PREV]]
// R21C-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]], // Rule ID 21 //
//
// R21O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R21O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R21N-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// R21N-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SELECT,
// R21N-NEXT:    // MIs[0] dst
// R21N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R21N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R21N-NEXT:    // MIs[0] src1
// R21N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R21N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R21N-NEXT:    // MIs[0] src2
// R21N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
//
// R21O-NEXT:    GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIPFP_MI_Predicate_frag,
// R21C-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex,
// R21N-NEXT:    // MIs[0] src3
// R21N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
// R21C-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/3, /*Renderer*/1, GICP_gi_complex,
// R21N-NEXT:    GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIPFP_MI_Predicate_frag,
// R21C-NEXT:    // (select:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src2, complex:{ *:[i32] }:$src3)<<P:Predicate_frag>> => (INSN2:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src3, complex:{ *:[i32] }:$src2)

// R21C-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN2,
// R21C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// R21C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// R21C-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/1,
// R21C-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/0,
// R21C-NEXT:    GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
// R21C-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// R21C-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// R21C-NEXT:    // GIR_Coverage, 21,
// R21C-NEXT:    GIR_Done,
// R21C-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
//
// R21O-NEXT:  GIM_Reject,
// R21O-NEXT:  // Label [[GROUP_NUM]]: @[[GROUP]]
// R21O-NEXT:  GIM_Reject,
// R21O:       // Label [[DEFAULT_NUM]]: @[[DEFAULT]]
// R21O-NEXT:  GIM_Reject,
// R21O-NEXT:  };

//===- Test a pattern with ComplexPattern operands. -----------------------===//
//
// R20O-NEXT:  GIM_SwitchOpcode, /*MI*/0, /*[*/{{[0-9]+}}, {{[0-9]+}}, /*)*//*default:*//*Label [[DEFAULT_NUM:[0-9]+]]*/ [[DEFAULT:[0-9]+]],
// R20O-NEXT:  /*TargetOpcode::G_ADD*//*Label [[CASE_ADD_NUM:[0-9]+]]*/ [[CASE_ADD:[0-9]+]],
// R20O:       /*TargetOpcode::G_SUB*//*Label [[CASE_SUB_NUM:[0-9]+]]*/ [[CASE_SUB:[0-9]+]],
// R20O:       // Label [[CASE_ADD_NUM]]: @[[CASE_ADD]]
// R20O:       // Label [[CASE_SUB_NUM]]: @[[CASE_SUB]]
// R20O-NEXT:  GIM_Try, /*On fail goto*//*Label [[GROUP_NUM:[0-9]+]]*/ [[GROUP:[0-9]+]],
// R20O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R20O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R20O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R20O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
//
// R20N:       GIM_Try, /*On fail goto*//*Label [[PREV_NUM:[0-9]+]]*/ [[PREV:[0-9]+]], // Rule ID 21 //
// R20N:       // Label [[PREV_NUM]]: @[[PREV]]
//
// R20C-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]], // Rule ID 20 //
//
// R20N-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// R20N-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SUB,
// R20N-NEXT:    // MIs[0] dst
// R20N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R20N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R20N-NEXT:    // MIs[0] src1
// R20N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
//
// R20N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R20N-NEXT:    // MIs[0] src2
// R20N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R20O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R20C-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex,
// R20C-NEXT:    // (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src2) => (INSN1:{ *:[i32] } GPR32:{ *:[i32] }:$src1, complex:{ *:[i32] }:$src2)
// R20C-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN1,
// R20C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// R20C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// R20C-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/0,
// R20C-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// R20C-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// R20C-NEXT:    // GIR_Coverage, 20,
// R20C-NEXT:    GIR_Done,
// R20C-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
//
// R20O:       // Label [[GROUP_NUM]]: @[[GROUP]]
// R20O-NEXT:  GIM_Reject,
// R20O:       // Label [[DEFAULT_NUM]]: @[[DEFAULT]]
// R20O-NEXT:  GIM_Reject,
// R20O-NEXT:  };

def INSN1 : I<(outs GPR32:$dst), (ins GPR32:$src1, complex:$src2), []>;
def : Pat<(sub GPR32:$src1, complex:$src2), (INSN1 GPR32:$src1, complex:$src2)>;

//===- Test a pattern with multiple ComplexPattern operands. --------------===//
//
def : GINodeEquiv<G_SELECT, select>;
let mayLoad = 1 in {
  def INSN2 : I<(outs GPR32:$dst), (ins GPR32Op:$src1, complex:$src2, complex:$src3), []>;
}
def frag : PatFrag<(ops node:$a, node:$b, node:$c),
                   (select node:$a, node:$b, node:$c),
                   [{ return true; // C++ code }]> {
  let GISelPredicateCode = [{ return true; // C++ code }];
}
def : Pat<(frag GPR32:$src1, complex:$src2, complex:$src3),
          (INSN2 GPR32:$src1, complex:$src3, complex:$src2)>;

//===- Test a more complex multi-instruction match. -----------------------===//
//
// R00O-NEXT:  GIM_SwitchOpcode, /*MI*/0, /*[*/{{[0-9]+}}, {{[0-9]+}}, /*)*//*default:*//*Label [[DEFAULT_NUM:[0-9]+]]*/ [[DEFAULT:[0-9]+]],
// R00O-NEXT:  /*TargetOpcode::G_ADD*//*Label [[CASE_ADD_NUM:[0-9]+]]*/ [[CASE_ADD:[0-9]+]],
// R00O:       /*TargetOpcode::G_SUB*//*Label [[CASE_SUB_NUM:[0-9]+]]*/ [[CASE_SUB:[0-9]+]],
// R00O:       // Label [[CASE_ADD_NUM]]: @[[CASE_ADD]]
// R00O:       // Label [[CASE_SUB_NUM]]: @[[CASE_SUB]]
// R00O-NEXT:  GIM_Try, /*On fail goto*//*Label [[GROUP_NUM:[0-9]+]]*/ [[GROUP:[0-9]+]],
// R00O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R00O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R00O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R00O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
//
// R00C:       GIM_Try, /*On fail goto*//*Label [[PREV_NUM:[0-9]+]]*/ [[PREV:[0-9]+]], // Rule ID 20 //
// R00C:       // Label [[PREV_NUM]]: @[[PREV]]
//
// R00C-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]], // Rule ID 0 //
// R00C-NEXT:    GIM_CheckFeatures, GIFBS_HasA,
// R00N-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// R00N-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SUB,
// R00N-NEXT:    // MIs[0] dst
// R00N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R00N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R00N-NEXT:    // MIs[0] Operand 1
// R00N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R00C-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// R00N-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// R00C-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
// R00N-NEXT:    // MIs[1] Operand 0
// R00N-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// R00N-NEXT:    // MIs[1] src1
// R00C-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// R00O-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// R00N-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R00N-NEXT:    // MIs[1] src2
// R00N-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// R00N-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// R00N-NEXT:    // MIs[0] Operand 2
// R00N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R00O-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R00O-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// R00C-NEXT:    GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// R00N-NEXT:    GIM_CheckNumOperands, /*MI*/2, /*Expected*/3,
// R00C-NEXT:    GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_SUB,
// R00N-NEXT:    // MIs[2] Operand 0
// R00N-NEXT:    GIM_CheckType, /*MI*/2, /*Op*/0, /*Type*/GILLT_s32,
// R00N-NEXT:    // MIs[2] src3
// R00C-NEXT:    GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
// R00O-NEXT:    GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
// R00N-NEXT:    GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R00N-NEXT:    // MIs[2] src4
// R00N-NEXT:    GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
// R00N-NEXT:    GIM_CheckRegBankForClass, /*MI*/2, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// R00O-NEXT:    GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R00O-NEXT:    GIM_CheckRegBankForClass, /*MI*/2, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// R00C-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// R00C-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/2,
// R00C-NEXT:    // (sub:{ *:[i32] } (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2), (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src3, GPR32:{ *:[i32] }:$src4)) => (INSNBOB:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src3, GPR32:{ *:[i32] }:$src4)
// R00C-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSNBOB,
// R00C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// R00C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// R00C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// R00C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src3
// R00C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // src4
// R00C-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// R00C-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// R00C-NEXT:    // GIR_Coverage, 0,
// R00C-NEXT:    GIR_Done,
// R00C-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
//
// R00O-NEXT:  GIM_Reject,
// R00O-NEXT:  // Label [[GROUP_NUM]]: @[[GROUP]]
// R00O-NEXT:  GIM_Reject,
// R00O:       // Label [[DEFAULT_NUM]]: @[[DEFAULT]]
// R00O-NEXT:  GIM_Reject,
// R00O-NEXT:  };

def INSNBOB : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3, GPR32:$src4),
                 [(set GPR32:$dst,
                      (sub (sub GPR32:$src1, GPR32:$src2), (sub GPR32:$src3, GPR32:$src4)))]>,
               Requires<[HasA]>;

//===- Test a simple pattern with an intrinsic. ---------------------------===//
//
// R01O-NEXT:  GIM_SwitchOpcode, /*MI*/0, /*[*/{{[0-9]+}}, {{[0-9]+}}, /*)*//*default:*//*Label [[DEFAULT_NUM:[0-9]+]]*/ [[DEFAULT:[0-9]+]],
// R01O-NEXT:  /*TargetOpcode::G_ADD*//*Label [[CASE_ADD_NUM:[0-9]+]]*/ [[CASE_ADD:[0-9]+]],
// R01O:       /*TargetOpcode::G_INTRINSIC*//*Label [[CASE_INTRINSIC_NUM:[0-9]+]]*/ [[CASE_INTRINSIC:[0-9]+]],
// R01O:       // Label [[CASE_ADD_NUM]]: @[[CASE_ADD]]
// R01O:       // Label [[CASE_INTRINSIC_NUM]]: @[[CASE_INTRINSIC]]
//
// R01N:       GIM_Try, /*On fail goto*//*Label [[PREV_NUM:[0-9]+]]*/ [[PREV:[0-9]+]], // Rule ID 0 //
// R01N:       // Label [[PREV_NUM]]: @[[PREV]]
//
// R01C-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]], // Rule ID 1 //
// R01C-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
//
// R01O-NEXT:    GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::mytarget_nop,
// R01O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R01O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R01O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
//
// R01N-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_INTRINSIC,
// R01N-NEXT:    // MIs[0] dst
// R01N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R01N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R01N-NEXT:    // MIs[0] Operand 1
// R01N-NEXT:    GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::mytarget_nop,
// R01N-NEXT:    // MIs[0] src1
// R01N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
//
// R01C-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// R01C-NEXT:    // (intrinsic_wo_chain:{ *:[i32] } [[ID:[0-9]+]]:{ *:[iPTR] }, GPR32:{ *:[i32] }:$src1) => (MOV:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// R01C-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOV,
// R01C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// R01C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
// R01C-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// R01C-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// R01C-NEXT:    // GIR_Coverage, 1,
// R01C-NEXT:    GIR_Done,
// R01C-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
//
// R01O-NEXT:  GIM_Reject,
// R01O:       // Label [[DEFAULT_NUM]]: @[[DEFAULT]]
// R01O-NEXT:  GIM_Reject,

def MOV : I<(outs GPR32:$dst), (ins GPR32:$src1),
            [(set GPR32:$dst, (int_mytarget_nop GPR32:$src1))]>;

//===- Test a simple pattern with a default operand. ----------------------===//
//
// R02O-NEXT:  GIM_SwitchOpcode, /*MI*/0, /*[*/{{[0-9]+}}, {{[0-9]+}}, /*)*//*default:*//*Label [[DEFAULT_NUM:[0-9]+]]*/ [[DEFAULT:[0-9]+]],
// R02O-NEXT:  /*TargetOpcode::G_ADD*//*Label [[CASE_ADD_NUM:[0-9]+]]*/ [[CASE_ADD:[0-9]+]],
// R02O:       /*TargetOpcode::G_XOR*//*Label [[CASE_XOR_NUM:[0-9]+]]*/ [[CASE_XOR:[0-9]+]],
// R02O:       // Label [[CASE_ADD_NUM]]: @[[CASE_ADD]]
// R02O:       // Label [[CASE_XOR_NUM]]: @[[CASE_XOR]]
// R02O-NEXT:  GIM_Try, /*On fail goto*//*Label [[GROUP_NUM:[0-9]+]]*/ [[GROUP:[0-9]+]],
// R02O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R02O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R02O-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// R02O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R02O-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
//
// R02N:       GIM_Try, /*On fail goto*//*Label [[PREV_NUM:[0-9]+]]*/ [[PREV:[0-9]+]], // Rule ID 1 //
// R02N:       // Label [[PREV_NUM]]: @[[PREV]]
//
// R02C-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]], // Rule ID 2 //
//
// R02N-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// R02N-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// R02N-NEXT:    // MIs[0] dst
// R02N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// R02N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// R02N-NEXT:    // MIs[0] src1
// R02N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// R02N-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// R02N-NEXT:    // MIs[0] Operand 2
// R02N-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
//
// R02C-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -2
// R02C-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -2:{ *:[i32] }) => (XORI:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// R02C-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORI,
// R02C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// R02C-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// R02C-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// R02C-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// R02C-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// R02C-NEXT:    // GIR_Coverage, 2,
// R02C-NEXT:    GIR_Done,
// R02C-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
//
// R02O:       // Label [[DEFAULT_NUM]]: @[[DEFAULT]]
// R02O-NEXT:  GIM_Reject,

// The -2 is just to distinguish it from the 'not' case below.
def XORI : I<(outs GPR32:$dst), (ins m1:$src2, GPR32:$src1),
             [(set GPR32:$dst, (xor GPR32:$src1, -2))]>;

//===- Test a simple pattern with a default register operand. -------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -3
// NOOPT-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -3:{ *:[i32] }) => (XOR:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XOR,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 3,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

// The -3 is just to distinguish it from the 'not' case below and the other default op case above.
def XOR : I<(outs GPR32:$dst), (ins Z:$src2, GPR32:$src1),
            [(set GPR32:$dst, (xor GPR32:$src1, -3))]>;

//===- Test a simple pattern with a multiple default operands. ------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -4
// NOOPT-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -4:{ *:[i32] }) => (XORlike:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORlike,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// NOOPT-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 4,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

// The -4 is just to distinguish it from the other 'not' cases.
def XORlike : I<(outs GPR32:$dst), (ins m1Z:$src2, GPR32:$src1),
                [(set GPR32:$dst, (xor GPR32:$src1, -4))]>;

//===- Test a simple pattern with multiple operands with defaults. --------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -5,
// NOOPT-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src1, -5:{ *:[i32] }) => (XORManyDefaults:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORManyDefaults,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// NOOPT-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// NOOPT-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 5,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

// The -5 is just to distinguish it from the other cases.
def XORManyDefaults : I<(outs GPR32:$dst), (ins m1Z:$src3, Z:$src2, GPR32:$src1),
                        [(set GPR32:$dst, (xor GPR32:$src1, -5))]>;

//===- Test a simple pattern with constant immediate operands. ------------===//
//
// This must precede the 3-register variants because constant immediates have
// priority over register banks.
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_XOR,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Wm
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
// NOOPT-NEXT:    // (xor:{ *:[i32] } GPR32:{ *:[i32] }:$Wm, -1:{ *:[i32] }) => (ORN:{ *:[i32] } R0:{ *:[i32] }, GPR32:{ *:[i32] }:$Wm)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::ORN,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // Wm
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 22,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def ORN : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def : Pat<(not GPR32:$Wm), (ORN R0, GPR32:$Wm)>;

//===- Test a nested instruction match. -----------------------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckFeatures, GIFBS_HasA,
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
// NOOPT-NEXT:    // MIs[1] Operand 0
// NOOPT-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    // MIs[1] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[1] src2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src3
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// NOOPT-NEXT:    // (mul:{ *:[i32] } (add:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2), GPR32:{ *:[i32] }:$src3)  =>  (MULADD:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src3)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MULADD,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 6,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

// We also get a second rule by commutativity.
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckFeatures, GIFBS_HasA,
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src3
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2,
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
// NOOPT-NEXT:    // MIs[1] Operand 0
// NOOPT-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    // MIs[1] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[1] src2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// NOOPT-NEXT:    // (mul:{ *:[i32] } GPR32:{ *:[i32] }:$src3, (add:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2))  =>  (MULADD:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src3)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MULADD,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 26,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def MULADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3),
               [(set GPR32:$dst,
                     (mul (add GPR32:$src1, GPR32:$src2), GPR32:$src3))]>,
             Requires<[HasA]>;

//===- Test a simple pattern with just a specific leaf immediate. ---------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 1
// NOOPT-NEXT:    GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 1,
// NOOPT-NEXT:    // 1:{ *:[i32] }  =>  (MOV1:{ *:[i32] })
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOV1,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 7,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def MOV1 : I<(outs GPR32:$dst), (ins), [(set GPR32:$dst, 1)]>;

//===- Test a simple pattern with a leaf immediate and a predicate. -------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// NOOPT-NEXT:    GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_simm8,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 1
// NOOPT-NEXT:    // No operand predicates
// NOOPT-NEXT:    // (imm:{ *:[i32] })<<P:Predicate_simm8>>:$imm => (MOVimm8:{ *:[i32] } (imm:{ *:[i32] }):$imm)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVimm8,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 8,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def simm8 : ImmLeaf<i32, [{ return isInt<8>(Imm); }]>;
def MOVimm8 : I<(outs GPR32:$dst), (ins i32imm:$imm), [(set GPR32:$dst, simm8:$imm)]>;

//===- Same again but use an IntImmLeaf. ----------------------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// NOOPT-NEXT:    GIM_CheckAPIntImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APInt_Predicate_simm9,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 1
// NOOPT-NEXT:    // No operand predicates
// NOOPT-NEXT:    // (imm:{ *:[i32] })<<P:Predicate_simm9>>:$imm =>  (MOVimm9:{ *:[i32] } (imm:{ *:[i32] }):$imm)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVimm9,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 9,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def simm9 : IntImmLeaf<i32, [{ return isInt<9>(Imm->getSExtValue()); }]>;
def MOVimm9 : I<(outs GPR32:$dst), (ins i32imm:$imm), [(set GPR32:$dst, simm9:$imm)]>;

//===- Test a pattern with a custom renderer. -----------------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// NOOPT-NEXT:    GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_cimm8,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 1
// NOOPT-NEXT:    // No operand predicates
// NOOPT-NEXT:    // (imm:{ *:[i32] })<<P:Predicate_cimm8>><<X:cimm8_xform>>:$imm  =>  (MOVcimm8:{ *:[i32] } (cimm8_xform:{ *:[i32] } (imm:{ *:[i32] }):$imm))
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVcimm8,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_CustomRenderer, /*InsnID*/0, /*OldInsnID*/0, /*Renderer*/GICR_renderImm8, // imm
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 10,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def MOVcimm8 : I<(outs GPR32:$dst), (ins i32imm:$imm), [(set GPR32:$dst, cimm8:$imm)]>;

//===- Test a simple pattern with a FP immediate and a predicate. ---------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FCONSTANT,
// NOOPT-NEXT:    GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimmz,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::FPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 1
// NOOPT-NEXT:    // No operand predicates
// NOOPT-NEXT:    // (fpimm:{ *:[f32] })<<P:Predicate_fpimmz>>:$imm =>  (MOVfpimmz:{ *:[f32] } (fpimm:{ *:[f32] }):$imm)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVfpimmz,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_CopyFConstantAsFPImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 17,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

//===- Test a simple pattern with inferred pointer operands. ---------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// NOOPT-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// NOOPT-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // (ld:{ *:[i32] } GPR32:{ *:[i32] }:$src1)<<P:Predicate_unindexedload>><<P:Predicate_load>> => (LOAD:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// NOOPT-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::LOAD,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 11,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def LOAD : I<(outs GPR32:$dst), (ins GPR32:$src1),
            [(set GPR32:$dst, (load GPR32:$src1))]>;

//===- Test a simple pattern with explicit pointer operands. ---------------===//

// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// NOOPT-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// NOOPT-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_p0s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src
// NOOPT-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // (ld:{ *:[i32] } GPR32:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD:{ *:[i32] } GPR32:{ *:[i32] }:$src)
// NOOPT-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::LOAD,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 23,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def : Pat<(load GPR32:$src),
          (p0 (LOAD GPR32:$src))>;

//===- Test a simple pattern with a sextload -------------------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SEXTLOAD,
// NOOPT-NEXT:    GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/2,
// NOOPT-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // (ld:{ *:[i32] } GPR32:{ *:[i32] }:$src1)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi16>>  =>  (SEXTLOAD:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// NOOPT-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::SEXTLOAD,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 12,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def SEXTLOAD : I<(outs GPR32:$dst), (ins GPR32:$src1),
                 [(set GPR32:$dst, (sextloadi16 GPR32:$src1))]>;

//===- Test a simple pattern with regclass operands. ----------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ADD,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID
// NOOPT-NEXT:    // MIs[0] src2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // (add:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2) => (ADD:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)
// NOOPT-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::ADD,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 13,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2),
            [(set GPR32:$dst, (add GPR32:$src1, GPR32:$src2))]>;

//===- Test a pattern with a tied operand in the matcher ------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ADD,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src{{$}}
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src{{$}}
// NOOPT-NEXT:    GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
// NOOPT-NEXT:    // (add:{ *:[i32] } GPR32:{ *:[i32] }:$src, GPR32:{ *:[i32] }:$src) => (DOUBLE:{ *:[i32] } GPR32:{ *:[i32] }:$src)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::DOUBLE,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 14,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def DOUBLE : I<(outs GPR32:$dst), (ins GPR32:$src), [(set GPR32:$dst, (add GPR32:$src, GPR32:$src))]>;

//===- Test a simple pattern with ValueType operands. ----------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ADD,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    // MIs[0] src2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    // (add:{ *:[i32] } i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2) => (ADD:{ *:[i32] } i32:{ *:[i32] }:$src1, i32:{ *:[i32] }:$src2)
// NOOPT-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::ADD,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 24,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def : Pat<(add i32:$src1, i32:$src2),
          (ADD i32:$src1, i32:$src2)>;

//===- Test another simple pattern with regclass operands. ----------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckFeatures, GIFBS_HasA_HasB_HasC,
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src2
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // (mul:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2) => (MUL:{ *:[i32] } GPR32:{ *:[i32] }:$src2, GPR32:{ *:[i32] }:$src1)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MUL,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 15,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def MUL : I<(outs GPR32:$dst), (ins GPR32:$src2, GPR32:$src1),
             [(set GPR32:$dst, (mul GPR32:$src1, GPR32:$src2))]>,
          Requires<[HasA, HasB, HasC]>;

//===- Test a COPY_TO_REGCLASS --------------------------------------------===//
//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_BITCAST,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] src1
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::FPR32RegClassID,
// NOOPT-NEXT:    // (bitconvert:{ *:[i32] } FPR32:{ *:[f32] }:$src1) => (COPY_TO_REGCLASS:{ *:[i32] } FPR32:{ *:[f32] }:$src1, GPR32:{ *:[i32] })
// NOOPT-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
// NOOPT-NEXT:    GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // GIR_Coverage, 25,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def : Pat<(i32 (bitconvert FPR32:$src1)),
          (COPY_TO_REGCLASS FPR32:$src1, GPR32)>;

//===- Test a simple pattern with just a leaf immediate. ------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// NOOPT-NEXT:    // MIs[0] dst
// NOOPT-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// NOOPT-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// NOOPT-NEXT:    // MIs[0] Operand 1
// NOOPT-NEXT:    // No operand predicates
// NOOPT-NEXT:    // (imm:{ *:[i32] }):$imm =>  (MOVimm:{ *:[i32] } (imm:{ *:[i32] }):$imm)
// NOOPT-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOVimm,
// NOOPT-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// NOOPT-NEXT:    GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// NOOPT-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 16,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def MOVimm : I<(outs GPR32:$dst), (ins i32imm:$imm), [(set GPR32:$dst, imm:$imm)]>;

def fpimmz : FPImmLeaf<f32, [{ return Imm->isExactlyValue(0.0); }]>;
def MOVfpimmz : I<(outs FPR32:$dst), (ins f32imm:$imm), [(set FPR32:$dst, fpimmz:$imm)]>;

//===- Test a pattern with an MBB operand. --------------------------------===//
//
// NOOPT-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// NOOPT-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/1,
// NOOPT-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_BR,
// NOOPT-NEXT:    // MIs[0] target
// NOOPT-NEXT:    GIM_CheckIsMBB, /*MI*/0, /*Op*/0,
// NOOPT-NEXT:    // (br (bb:{ *:[Other] }):$target) => (BR (bb:{ *:[Other] }):$target)
// NOOPT-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::BR,
// NOOPT-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// NOOPT-NEXT:    // GIR_Coverage, 18,
// NOOPT-NEXT:    GIR_Done,
// NOOPT-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def BR : I<(outs), (ins unknown:$target),
            [(br bb:$target)]>;

// NOOPT-NEXT:    GIM_Reject,
// NOOPT-NEXT:  };
// NOOPT-NEXT:  return MatchTable0;
