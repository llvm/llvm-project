// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common -optimize-match-table=false %s -o %T/hwmode-non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK -input-file=%T/hwmode-non-optimized.cpp

include "llvm/Target/Target.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

class MyTargetGenericInstruction : GenericInstruction {
  let Namespace = "MyTarget";
}

//def Has32 : Predicate<"Subtarget->has32()">;
def Has64 : Predicate<"Subtarget->has64()">;

//def Mode32 : HwMode<"+a", [Has32]>;
def Mode64 : HwMode<"+b", [Has64]>;

def ModeVT : ValueTypeByHwMode<[DefaultMode, Mode64],
                               [i32,  i64]>;
def ModeRI : RegInfoByHwMode<
      [DefaultMode,       Mode64],
      [RegInfo<32,32,32>, RegInfo<64,64,64>]>;

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR : RegisterClass<"MyTarget", [ModeVT], 32, (add R0)> {
  let RegInfos = ModeRI;
}

def p0 : PtrValueTypeByHwMode<ModeVT, 0>;

class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

//===- Test the function boilerplate. -------------------------------------===//

// CHECK: const unsigned MAX_SUBTARGET_PREDICATES = 2;
// CHECK: using PredicateBitset = llvm::Bitset<MAX_SUBTARGET_PREDICATES>;

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_DECL
// CHECK-NEXT:    mutable MatcherState State;
// CHECK-NEXT:    typedef ComplexRendererFns(MyTargetInstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;
// CHECK-NEXT:    typedef void(MyTargetInstructionSelector::*CustomRendererFn)(MachineInstrBuilder &, const MachineInstr &, int) const;
// CHECK-NEXT:    const ExecInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn> ExecInfo;
// CHECK-NEXT:    static MyTargetInstructionSelector::ComplexMatcherMemFn ComplexPredicateFns[];
// CHECK-NEXT:    static MyTargetInstructionSelector::CustomRendererFn CustomRenderers[];
// CHECK-NEXT:    bool testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const override;
// CHECK-NEXT:    bool testImmPredicate_APInt(unsigned PredicateID, const APInt &Imm) const override;
// CHECK-NEXT:    bool testImmPredicate_APFloat(unsigned PredicateID, const APFloat &Imm) const override;
// CHECK-NEXT:    const int64_t *getMatchTable() const override;
// CHECK-NEXT:    bool testMIPredicate_MI(unsigned PredicateID, const MachineInstr &MI, const MatcherState &State) const override;
// CHECK-NEXT:    bool testSimplePredicate(unsigned PredicateID) const override;
// CHECK-NEXT:    void runCustomAction(unsigned FnID, const MatcherState &State, NewMIVector &OutMIs) const override;
// CHECK-NEXT:  #endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_INIT
// CHECK-NEXT:    , State(0),
// CHECK-NEXT:    ExecInfo(TypeObjects, NumTypeObjects, FeatureBitsets, ComplexPredicateFns, CustomRenderers)
// CHECK-NEXT:  #endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT

// CHECK-LABEL: // LLT Objects.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GILLT_p0s32,
// CHECK-NEXT:    GILLT_p0s64,
// CHECK-NEXT:    GILLT_s32,
// CHECK-NEXT:    GILLT_s64,
// CHECK-NEXT:  }
// CHECK-NEXT:  const static size_t NumTypeObjects = 4;
// CHECK-NEXT:  const static LLT TypeObjects[] = {
// CHECK-NEXT:    LLT::pointer(0, 32),
// CHECK-NEXT:    LLT::pointer(0, 64),
// CHECK-NEXT:    LLT::scalar(32),
// CHECK-NEXT:    LLT::scalar(64),
// CHECK-NEXT:  };

// CHECK-LABEL: enum SubtargetFeatureBits : uint8_t {
// CHECK-NEXT:    Feature_HwMode1Bit = 1,
// CHECK-NEXT:    Feature_HwMode0Bit = 0,
// CHECK-NEXT:  };

// CHECK-LABEL: PredicateBitset MyTargetInstructionSelector::
// CHECK-NEXT:  computeAvailableModuleFeatures(const MyTargetSubtarget *Subtarget) const {
// CHECK-NEXT:    PredicateBitset Features;
// CHECK-NEXT:    if (!((Subtarget->has64())))
// CHECK-NEXT:      Features.set(Feature_HwMode1Bit);
// CHECK-NEXT:    if ((Subtarget->has64()))
// CHECK-NEXT:      Features.set(Feature_HwMode0Bit);
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: PredicateBitset MyTargetInstructionSelector::
// CHECK-NEXT:  computeAvailableFunctionFeatures(const MyTargetSubtarget *Subtarget, const MachineFunction *MF) const {
// CHECK-NEXT:    PredicateBitset Features;
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: // Feature bitsets.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIFBS_Invalid,
// CHECK-NEXT:    GIFBS_HwMode0,
// CHECK-NEXT:    GIFBS_HwMode1,
// CHECK-NEXT:  }
// CHECK-NEXT:  constexpr static PredicateBitset FeatureBitsets[] {
// CHECK-NEXT:    {}, // GIFBS_Invalid
// CHECK-NEXT:    {Feature_HwMode0Bit, },
// CHECK-NEXT:    {Feature_HwMode1Bit, },
// CHECK-NEXT:  };

// CHECK: bool MyTargetInstructionSelector::selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const {
// CHECK-NEXT: const PredicateBitset AvailableFeatures = getAvailableFeatures();
// CHECK-NEXT: MachineIRBuilder B(I);
// CHECK-NEXT: State.MIs.clear();
// CHECK-NEXT: State.MIs.push_back(&I);

// CHECK:      if (executeMatchTable(*this, State, ExecInfo, B, getMatchTable(), TII, MF->getRegInfo(), TRI, RBI, AvailableFeatures, &CoverageInfo)) {
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

// CHECK: const int64_t *
// CHECK-LABEL: MyTargetInstructionSelector::getMatchTable() const {
// CHECK-NEXT: MatchTable0[] = {

//===- Test a simple pattern with inferred pointer operands. ---------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HwMode0,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// CHECK-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/64,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // (ld:{ *:[i64] } GPR:{ *:[i64] }:$src1)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD:{ *:[i64] } GPR:{ *:[i64] }:$src1)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::LOAD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    // GIR_Coverage, 0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HwMode1,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// CHECK-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // (ld:{ *:[i32] } GPR:{ *:[i32] }:$src1)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD:{ *:[i32] } GPR:{ *:[i32] }:$src1)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::LOAD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    // GIR_Coverage, 1,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def LOAD : I<(outs GPR:$dst), (ins GPR:$src1),
            [(set GPR:$dst, (load GPR:$src1))]>;

//===- Test a simple pattern with explicit pointer operands. ---------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HwMode0,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// CHECK-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_p0s64,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // MIs[0] src
// CHECK-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/64,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // (ld:{ *:[i64] } GPR:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD:{ *:[i64] } GPR:{ *:[i64] }:$src)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::LOAD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    // GIR_Coverage, 2,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ [[LABEL:[0-9]+]],
// CHECK-NEXT:    GIM_CheckFeatures, GIFBS_HwMode1,
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// CHECK-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_p0s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // MIs[0] src
// CHECK-NEXT:    GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::GPRRegClassID,
// CHECK-NEXT:    // (ld:{ *:[i32] } GPR:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD:{ *:[i32] } GPR:{ *:[i32] }:$src)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/MyTarget::LOAD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    // GIR_Coverage, 3,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]

def : Pat<(load GPR:$src),
          (p0 (LOAD GPR:$src))>;
