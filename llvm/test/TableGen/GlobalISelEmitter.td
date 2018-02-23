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
// CHECK-NEXT:    typedef ComplexRendererFn(MyTargetInstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;
// CHECK-NEXT:    const MatcherInfoTy<PredicateBitset, ComplexMatcherMemFn> MatcherInfo;
// CHECK-NEXT:  #endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL

// CHECK-LABEL: #ifdef GET_GLOBALISEL_TEMPORARIES_INIT
// CHECK-NEXT:    , State(2),
// CHECK-NEXT:    MatcherInfo({TypeObjects, FeatureBitsets, {
// CHECK-NEXT:      nullptr, // GICP_Invalid
// CHECK-NEXT:      &MyTargetInstructionSelector::selectComplexPattern, // gi_complex
// CHECK-NEXT:    }})
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
// CHECK-NEXT:    // (select:i32 GPR32:i32:$src1, complex:i32:$src2, complex:i32:$src3) => (INSN2:i32 GPR32:i32:$src1, complex:i32:$src3, complex:i32:$src2)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN2,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/1,
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/0,
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable0\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable0, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def INSN3 : I<(outs GPR32:$dst),
              (ins GPR32Op:$src1, complex:$src2, GPR32:$src3, complex:$src4, complex:$src5), []>;
def : Pat<(select GPR32:$src1, complex:$src2, (select GPR32:$src3, complex:$src4, complex:$src5)),
          (INSN3 GPR32:$src1, complex:$src2, GPR32:$src3, complex:$src4, complex:$src5)>;

//===- Test a pattern with multiple ComplexPattern operands. --------------===//
//

// CHECK-LABEL: MatchTable1[] = {
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
// CHECK-NEXT:    // MIs[0] src2
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/0, /*Op*/2, /*Renderer*/0, GICP_gi_complex,
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
// CHECK-NEXT:    // MIs[1] src5
// CHECK-NEXT:    GIM_CheckType, /*MI*/1, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckComplexPattern, /*MI*/1, /*Op*/3, /*Renderer*/2, GICP_gi_complex,
// CHECK-NEXT:    GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:    // (select:i32 GPR32:i32:$src1, complex:i32:$src2, (select:i32 GPR32:i32:$src3, complex:i32:$src4, complex:i32:$src5))  =>  (INSN3:i32 GPR32:i32:$src1, complex:i32:$src2, GPR32:i32:$src3, complex:i32:$src4, complex:i32:$src5)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN3,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src3
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/1,
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/2,
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable1\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable1, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def : GINodeEquiv<G_SELECT, select>;
def INSN2 : I<(outs GPR32:$dst), (ins GPR32Op:$src1, complex:$src2, complex:$src3), []>;
def : Pat<(select GPR32:$src1, complex:$src2, complex:$src3),
          (INSN2 GPR32:$src1, complex:$src3, complex:$src2)>;

//===- Test a simple pattern with regclass operands. ----------------------===//

// CHECK-LABEL: MatchTable2[] = {
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
// CHECK-NEXT:    // (add:i32 GPR32:i32:$src1, GPR32:i32:$src2) => (ADD:i32 GPR32:i32:$src1, GPR32:i32:$src2)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/ 0, /*Opcode*/MyTarget::ADD,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable2\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable2, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2),
            [(set GPR32:$dst, (add GPR32:$src1, GPR32:$src2))]>;

//===- Test a simple pattern with an intrinsic. ---------------------------===//
//

// CHECK-LABEL: MatchTable3[] = {
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
// CHECK-NEXT:    // (intrinsic_wo_chain:i32 [[ID:[0-9]+]]:iPTR, GPR32:i32:$src1) => (MOV:i32 GPR32:i32:$src1)

// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOV,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable3\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable3, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def MOV : I<(outs GPR32:$dst), (ins GPR32:$src1),
            [(set GPR32:$dst, (int_mytarget_nop GPR32:$src1))]>;

//===- Test a nested instruction match. -----------------------------------===//

// CHECK-LABEL: MatchTable4[] = {
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
// CHECK-NEXT:    // (mul:i32 (add:i32 GPR32:i32:$src1, GPR32:i32:$src2), GPR32:i32:$src3)  =>  (MULADD:i32 GPR32:i32:$src1, GPR32:i32:$src2, GPR32:i32:$src3)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MULADD,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable4\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable4, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

// We also get a second rule by commutativity.
// CHECK-LABEL: MatchTable5[] = {
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
// CHECK-NEXT:    // (mul:i32 GPR32:i32:$src3, (add:i32 GPR32:i32:$src1, GPR32:i32:$src2))  =>  (MULADD:i32 GPR32:i32:$src1, GPR32:i32:$src2, GPR32:i32:$src3)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MULADD,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable5\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable5, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def MULADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3),
               [(set GPR32:$dst,
                     (mul (add GPR32:$src1, GPR32:$src2), GPR32:$src3))]>,
             Requires<[HasA]>;

//===- Test another simple pattern with regclass operands. ----------------===//

// CHECK-LABEL: MatchTable6[] = {
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
// CHECK-NEXT:    // (mul:i32 GPR32:i32:$src1, GPR32:i32:$src2) => (MUL:i32 GPR32:i32:$src2, GPR32:i32:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MUL,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable6\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable6, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def MUL : I<(outs GPR32:$dst), (ins GPR32:$src2, GPR32:$src1),
             [(set GPR32:$dst, (mul GPR32:$src1, GPR32:$src2))]>,
          Requires<[HasA, HasB, HasC]>;

//===- Test a more complex multi-instruction match. -----------------------===//

// CHECK-LABEL: MatchTable7[] = {
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
// CHECK-NEXT:    // (sub:i32 (sub:i32 GPR32:i32:$src1, GPR32:i32:$src2), (sub:i32 GPR32:i32:$src3, GPR32:i32:$src4)) => (INSNBOB:i32 GPR32:i32:$src1, GPR32:i32:$src2, GPR32:i32:$src3, GPR32:i32:$src4)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSNBOB,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src3
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // src4
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable7\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable7, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def INSNBOB : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3, GPR32:$src4),
                 [(set GPR32:$dst,
                      (sub (sub GPR32:$src1, GPR32:$src2), (sub GPR32:$src3, GPR32:$src4)))]>,
               Requires<[HasA]>;

//===- Test a pattern with ComplexPattern operands. -----------------------===//
//

// CHECK-LABEL: MatchTable8[] = {
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
// CHECK-NEXT:    // (sub:i32 GPR32:i32:$src1, complex:i32:$src2) => (INSN1:i32 GPR32:i32:$src1, complex:i32:$src2)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::INSN1,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_ComplexRenderer, /*InsnID*/0, /*RendererID*/0,
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable8\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable8, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def INSN1 : I<(outs GPR32:$dst), (ins GPR32:$src1, complex:$src2), []>;
def : Pat<(sub GPR32:$src1, complex:$src2), (INSN1 GPR32:$src1, complex:$src2)>;

//===- Test a simple pattern with a default operand. ----------------------===//
//

// CHECK-LABEL: MatchTable9[] = {
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
// CHECK-NEXT:    // (xor:i32 GPR32:i32:$src1, -2:i32) => (XORI:i32 GPR32:i32:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORI,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable9\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable9, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

// The -2 is just to distinguish it from the 'not' case below.
def XORI : I<(outs GPR32:$dst), (ins m1:$src2, GPR32:$src1),
             [(set GPR32:$dst, (xor GPR32:$src1, -2))]>;

//===- Test a simple pattern with a default register operand. -------------===//
//

// CHECK-LABEL: MatchTable10[] = {
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
// CHECK-NEXT:    // (xor:i32 GPR32:i32:$src1, -3:i32) => (XOR:i32 GPR32:i32:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XOR,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable10\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable10, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

// The -3 is just to distinguish it from the 'not' case below and the other default op case above.
def XOR : I<(outs GPR32:$dst), (ins Z:$src2, GPR32:$src1),
            [(set GPR32:$dst, (xor GPR32:$src1, -3))]>;

//===- Test a simple pattern with a multiple default operands. ------------===//
//

// CHECK-LABEL: MatchTable11[] = {
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
// CHECK-NEXT:    // (xor:i32 GPR32:i32:$src1, -4:i32) => (XORlike:i32 GPR32:i32:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORlike,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable11\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable11, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

// The -4 is just to distinguish it from the other 'not' cases.
def XORlike : I<(outs GPR32:$dst), (ins m1Z:$src2, GPR32:$src1),
                [(set GPR32:$dst, (xor GPR32:$src1, -4))]>;

//===- Test a simple pattern with multiple operands with defaults. --------===//
//

// CHECK-LABEL: MatchTable12[] = {
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
// CHECK-NEXT:    // (xor:i32 GPR32:i32:$src1, -5:i32) => (XORManyDefaults:i32 GPR32:i32:$src1)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::XORManyDefaults,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddImm, /*InsnID*/0, /*Imm*/-1,
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable12\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable12, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

// The -5 is just to distinguish it from the other cases.
def XORManyDefaults : I<(outs GPR32:$dst), (ins m1Z:$src3, Z:$src2, GPR32:$src1),
                        [(set GPR32:$dst, (xor GPR32:$src1, -5))]>;

//===- Test a simple pattern with constant immediate operands. ------------===//
//
// This must precede the 3-register variants because constant immediates have
// priority over register banks.

// CHECK-LABEL: MatchTable13[] = {
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
// CHECK-NEXT:    // (xor:i32 GPR32:i32:$Wm, -1:i32) => (ORN:i32 R0:i32, GPR32:i32:$Wm)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::ORN,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_AddRegister, /*InsnID*/0, MyTarget::R0,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // Wm
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable13\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable13, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def ORN : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def : Pat<(not GPR32:$Wm), (ORN R0, GPR32:$Wm)>;

//===- Test a COPY_TO_REGCLASS --------------------------------------------===//
//

// CHECK-LABEL: MatchTable14[] = {
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_BITCAST,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] src1
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/MyTarget::FPR32RegClassID,
// CHECK-NEXT:    // (bitconvert:i32 FPR32:f32:$src1) => (COPY_TO_REGCLASS:i32 FPR32:f32:$src1, GPR32:i32)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/ 0, /*Opcode*/TargetOpcode::COPY,
// CHECK-NEXT:    GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GPR32*/ 1,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable14\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable14, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def : Pat<(i32 (bitconvert FPR32:$src1)),
          (COPY_TO_REGCLASS FPR32:$src1, GPR32)>;

//===- Test a simple pattern with just a leaf immediate. ------------------===//

// CHECK-LABEL: MatchTable15[] = {
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_CONSTANT,
// CHECK-NEXT:    // MIs[0] dst
// CHECK-NEXT:    GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:    GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT:    // MIs[0] Operand 1
// CHECK-NEXT:    GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 1,
// CHECK-NEXT:    // 1:i32  =>  (MOV1:i32)
// CHECK-NEXT:    GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::MOV1,
// CHECK-NEXT:    GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:    GIR_MergeMemOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable15\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable15, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def MOV1 : I<(outs GPR32:$dst), (ins), [(set GPR32:$dst, 1)]>;

//===- Test a pattern with an MBB operand. --------------------------------===//

// CHECK-LABEL: MatchTable16[] = {
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/1,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_BR,
// CHECK-NEXT:    // MIs[0] target
// CHECK-NEXT:    GIM_CheckIsMBB, /*MI*/0, /*Op*/0,
// CHECK-NEXT:    // (br (bb:Other):$target) => (BR (bb:Other):$target)
// CHECK-NEXT:    GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/ 0, /*Opcode*/MyTarget::BR,
// CHECK-NEXT:    GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:    GIR_Done,
// CHECK-NEXT:  };
// CHECK-NEXT:  MIs.resize(1);
// CHECK-NEXT:  DEBUG(dbgs() << "Processing MatchTable16\n");
// CHECK-NEXT:  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, MatchTable16, TII, MRI, TRI, RBI, AvailableFeatures)) {
// CHECK-NEXT:    return true;
// CHECK-NEXT:  }

def BR : I<(outs), (ins unknown:$target),
            [(br bb:$target)]>;
