// RUN: llvm-tblgen -gen-global-isel -I %p/../../../include -I %p/../Common -optimize-match-table=false %s -o %t.hwmode-non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK -input-file=%t.hwmode-non-optimized.cpp

include "llvm/Target/Target.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

class MyTargetGenericInstruction : GenericInstruction {
  let Namespace = "MyTarget";
}

//def Has32 : Predicate<"Subtarget->has32()">;
def Has64 : Predicate<"Subtarget->has64()">;

//def Mode32 : HwMode<[Has32]>;
def Mode64 : HwMode<[Has64]>;

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

def undef_tied : OperandWithDefaultOps<ModeVT, (ops (ModeVT undef_tied_input))> {
  let MIOperandInfo = (ops GPR:$inactive);
}

//===- Test the function boilerplate. -------------------------------------===//

// Only the pattern that is only valid in Mode64 needs a HwMode predicate.
// CHECK: const unsigned MAX_SUBTARGET_PREDICATES = 1;
// CHECK: using PredicateBitset = llvm::Bitset<MAX_SUBTARGET_PREDICATES>;

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_DECL
// CHECK-EMPTY:
// CHECK-NEXT:    mutable MatcherState State;
// CHECK-NEXT:    typedef ComplexRendererFns(MyTargetInstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;
// CHECK-NEXT:    typedef void(MyTargetInstructionSelector::*CustomRendererFn)(MachineInstrBuilder &, const MachineInstr &, int) const;
// CHECK-NEXT:    const ExecInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn> ExecInfo;
// CHECK-NEXT:    static MyTargetInstructionSelector::ComplexMatcherMemFn ComplexPredicateFns[];
// CHECK-NEXT:    static MyTargetInstructionSelector::CustomRendererFn CustomRenderers[];
// CHECK-NEXT:    bool testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const override;
// CHECK-NEXT:    bool testImmPredicate_APInt(unsigned PredicateID, const APInt &Imm) const override;
// CHECK-NEXT:    bool testImmPredicate_APFloat(unsigned PredicateID, const APFloat &Imm) const override;
// CHECK-NEXT:    const uint8_t *getMatchTable() const override;
// CHECK-NEXT:    LLT getLLTForHwMode(unsigned Index) const override;
// CHECK-NEXT:    bool testMIPredicate_MI(unsigned PredicateID, const MachineInstr &MI, const MatcherState &State) const override;
// CHECK-NEXT:    bool testMOPredicate_MO(unsigned PredicateID, const MachineOperand &MO, const MatcherState &State) const override;
// CHECK-NEXT:    bool testSimplePredicate(unsigned PredicateID) const override;
// CHECK-NEXT:    bool runCustomAction(unsigned FnID, const MatcherState &State, NewMIVector &OutMIs) const override;
// CHECK-EMPTY:
// CHECK-NEXT:  #endif // GET_GLOBALISEL_TEMPORARIES_DECL

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_INIT
// CHECK-EMPTY:
// CHECK-NEXT:    , State(0),
// CHECK-NEXT:    ExecInfo(TypeObjects, NumTypeObjects, FeatureBitsets, ComplexPredicateFns, CustomRenderers)
// CHECK-EMPTY:
// CHECK-NEXT:  #endif // GET_GLOBALISEL_TEMPORARIES_INIT

// Types that depend on the HwMode are not in the type objects.
// CHECK-LABEL: // LLT Objects.
// CHECK-NEXT:  enum {
// CHECK-NEXT:  };
// CHECK-NEXT:  const static size_t NumTypeObjects = 0;
// CHECK-NEXT:  const static LLT TypeObjects[] = {
// CHECK-NEXT:  };

// CHECK-LABEL: enum SubtargetFeatureBits : uint8_t {
// CHECK-NEXT:    Feature_HwMode0Bit = 0,
// CHECK-NEXT:  };

// CHECK-LABEL: PredicateBitset MyTargetInstructionSelector::
// CHECK-NEXT:  computeAvailableModuleFeatures(const MyTargetSubtarget *Subtarget) const {
// CHECK-NEXT:    PredicateBitset Features{};
// CHECK-NEXT:    if (((Subtarget->has64())))
// CHECK-NEXT:      Features.set(Feature_HwMode0Bit);
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: PredicateBitset MyTargetInstructionSelector::
// CHECK-NEXT:  computeAvailableFunctionFeatures(const MyTargetSubtarget *Subtarget, const MachineFunction *MF) const {
// CHECK-NEXT:    PredicateBitset Features{};
// CHECK-NEXT:    return Features;
// CHECK-NEXT:  }

// CHECK-LABEL: // Feature bitsets.
// CHECK-NEXT:  enum {
// CHECK-NEXT:    GIFBS_Invalid,
// CHECK-NEXT:    GIFBS_HwMode0,
// CHECK-NEXT:  }
// CHECK-NEXT:  constexpr static PredicateBitset FeatureBitsets[] {
// CHECK-NEXT:    {}, // GIFBS_Invalid
// CHECK-NEXT:    {Feature_HwMode0Bit, },
// CHECK-NEXT:  };

// CHECK: bool MyTargetInstructionSelector::selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const {
// CHECK-NEXT: const PredicateBitset AvailableFeatures = getAvailableFeatures();
// CHECK-NEXT: MachineIRBuilder B(I);
// CHECK-NEXT: State.MIs.clear();
// CHECK-NEXT: State.MIs.push_back(&I);

// CHECK:      if (executeMatchTable(*this, State, ExecInfo, B, getMatchTable(), TII, MF->getRegInfo(), TRI, RBI, AvailableFeatures, &CoverageInfo)) {
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

// CHECK: const uint8_t *
// CHECK-LABEL: MyTargetInstructionSelector::getMatchTable() const {
// CHECK-NEXT: MatchTable0[] = {

//===- Test a simple pattern with inferred pointer operands. ---------------===//

// The pointer type depends on the HwMode, so the pointer operand is checked
// with GIM_CheckPointerToAnyByHwMode.
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4([[L0:[0-9]+]]), // Rule ID 0 //
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_LOAD),
// CHECK-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:    // MIs[0] DstI[dst]
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/0, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckPointerToAnyByHwMode, /*MI*/0, /*Op*/1, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // (ld:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src1)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src1)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD),
// CHECK-NEXT:    GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:    // GIR_Coverage, 0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 0: @[[L0]]

def LOAD : I<(outs GPR:$dst), (ins GPR:$src1),
            [(set GPR:$dst, (load GPR:$src1))]>;

//===- Test a simple pattern with explicit pointer operands. ---------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4([[L1:[0-9]+]]), // Rule ID 1 //
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_LOAD),
// CHECK-NEXT:    GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:    GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:    // MIs[0] DstI[dst]
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/0, /*TypeByHwMode*/1,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // MIs[0] src
// CHECK-NEXT:    GIM_CheckPointerToAnyByHwMode, /*MI*/0, /*Op*/1, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // (ld:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD),
// CHECK-NEXT:    GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:    // GIR_Coverage, 1,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 1: @[[L1]]

def : Pat<(load GPR:$src),
          (p0 (LOAD GPR:$src))>;

//===- Test a dead def with a HwMode dependent type. ----------------------===//

// The dead def needs a temporary register created with GIR_MakeTempRegByHwMode.
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4([[L2:[0-9]+]]), // Rule ID 2 //
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ABS),
// CHECK-NEXT:    // MIs[0] DstI[dst]
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/0, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // MIs[0] src
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/1, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // (abs:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src)  =>  (TWO_DEFS:{ *:[i32] m1:[i64] }:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src)
// CHECK-NEXT:    GIR_MakeTempRegByHwMode, /*TempRegID*/0, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::TWO_DEFS),
// CHECK-NEXT:    GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:    GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define|RegState::Dead)),
// CHECK-NEXT:    GIR_RootToRootCopy, /*OpIdx*/1, // src
// CHECK-NEXT:    GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:    // GIR_Coverage, 2,
// CHECK-NEXT:    GIR_EraseRootFromParent_Done,
// CHECK-NEXT:  // Label 2: @[[L2]]

def TWO_DEFS : I<(outs GPR:$dst, GPR:$dead), (ins GPR:$src), []>;

def : Pat<(ModeVT (abs GPR:$src)), (TWO_DEFS GPR:$src)>;

//===- Test a pattern only valid in one HwMode. ---------------------------===//

// GPR only has an i64 type in Mode64 so this pattern is only valid in Mode64.
// It needs a HwMode predicate.
// CHECK-NEXT:  GIM_Try_CheckFeatures, /*On fail goto*//*Label 3*/ GIMT_Encode4([[L3:[0-9]+]]), GIMT_Encode2(GIFBS_HwMode0), // Rule ID 3 //
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_CTLZ),
// CHECK-NEXT:    // MIs[0] DstI[dst]
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/0, /*TypeByHwMode*/2,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // MIs[0] src
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/1, /*TypeByHwMode*/2,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // (ctlz:{ *:[] m1:[i64] } GPR:{ *:[] m1:[i64] }:$src)  =>  (LOAD:{ *:[] m1:[i64] } GPR:{ *:[] m1:[i64] }:$src)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD),
// CHECK-NEXT:    GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:    // GIR_Coverage, 3,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  // Label 3: @[[L3]]

def : Pat<(i64 (ctlz GPR:$src)), (LOAD GPR:$src)>;

//===- Test undef_tied_input with a HwMode dependent type. ----------------===//

// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4([[L4:[0-9]+]]), // Rule ID 4 //
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_CTPOP),
// CHECK-NEXT:    // MIs[0] DstI[dst]
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/0, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // MIs[0] src
// CHECK-NEXT:    GIM_RootCheckTypeByHwMode, /*Op*/1, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT:    // (ctpop:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src)  =>  (TIED:{ *:[i32] m1:[i64] } GPR:{ *:[i32] m1:[i64] }:$src)
// CHECK-NEXT:    GIR_MakeTempRegByHwMode, /*TempRegID*/0, /*TypeByHwMode*/0,
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::IMPLICIT_DEF),
// CHECK-NEXT:    GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT:    GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::TIED),
// CHECK-NEXT:    GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:    GIR_RootToRootCopy, /*OpIdx*/1, // src
// CHECK-NEXT:    GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:    GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:    // GIR_Coverage, 4,
// CHECK-NEXT:    GIR_EraseRootFromParent_Done,
// CHECK-NEXT:  // Label 4: @[[L4]]
// CHECK-NEXT:  GIM_Reject,
// CHECK-NEXT:  }; // Size: 211 bytes

def TIED : I<(outs GPR:$dst), (ins GPR:$src, undef_tied:$opt), []>;

def : Pat<(ModeVT (ctpop GPR:$src)), (TIED GPR:$src)>;

//===- Test the getLLTForHwMode function. ---------------------------------===//

// CHECK-LABEL: LLT MyTargetInstructionSelector::getLLTForHwMode(unsigned Index) const {
// CHECK-NEXT:   switch (Index) {
// CHECK-NEXT:   default:
// CHECK-NEXT:     llvm_unreachable("Unexpected index");
// CHECK-NEXT:   case 0:
// CHECK-NEXT:     switch (HwMode) {
// CHECK-NEXT:     default:
// CHECK-NEXT:       return LLT::scalar(32);
// CHECK-NEXT:     case 1:
// CHECK-NEXT:       return LLT::scalar(64);
// CHECK-NEXT:     }
// CHECK-NEXT:   case 1:
// CHECK-NEXT:     switch (HwMode) {
// CHECK-NEXT:     default:
// CHECK-NEXT:       return LLT::pointer(0, 32);
// CHECK-NEXT:     case 1:
// CHECK-NEXT:       return LLT::pointer(0, 64);
// CHECK-NEXT:     }
// CHECK-NEXT:   case 2:
// CHECK-NEXT:     switch (HwMode) {
// CHECK-NEXT:     default:
// CHECK-NEXT:       return LLT();
// CHECK-NEXT:     case 1:
// CHECK-NEXT:       return LLT::scalar(64);
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
