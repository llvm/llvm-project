//===-- SPIRVPostLegalizer.cpp - ammend info after legalization -*- C++ -*-===//
//
// which may appear after the legalizer pass
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass partially apply pre-legalization logic to new instructions inserted
// as a result of legalization:
// - assigns SPIR-V types to registers for new instructions.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Support/Debug.h"
#include <stack>

#define DEBUG_TYPE "spirv-postlegalizer"

using namespace llvm;

namespace {
class SPIRVPostLegalizer : public MachineFunctionPass {
public:
  static char ID;
  SPIRVPostLegalizer() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

namespace llvm {
//  Defined in SPIRVPreLegalizer.cpp.
extern void insertAssignInstr(Register Reg, Type *Ty, SPIRVType *SpirvTy,
                              SPIRVGlobalRegistry *GR, MachineIRBuilder &MIB,
                              MachineRegisterInfo &MRI);
extern void processInstr(MachineInstr &MI, MachineIRBuilder &MIB,
                         MachineRegisterInfo &MRI, SPIRVGlobalRegistry *GR,
                         SPIRVType *KnownResType);
} // namespace llvm

static bool mayBeInserted(unsigned Opcode) {
  switch (Opcode) {
  case TargetOpcode::G_CONSTANT:
  case TargetOpcode::G_UNMERGE_VALUES:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMAX:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMAXIMUM:
  case TargetOpcode::G_IMPLICIT_DEF:
  case TargetOpcode::G_BUILD_VECTOR:
  case TargetOpcode::G_ICMP:
  case TargetOpcode::G_SHUFFLE_VECTOR:
  case TargetOpcode::G_ANYEXT:
    return true;
  default:
    return isTypeFoldingSupported(Opcode);
  }
}

static SPIRVType *deduceTypeForGConstant(MachineInstr *I, MachineFunction &MF,
                                         SPIRVGlobalRegistry *GR,
                                         MachineIRBuilder &MIB,
                                         Register ResVReg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const LLT &Ty = MRI.getType(ResVReg);
  unsigned BitWidth = Ty.getScalarSizeInBits();
  return GR->getOrCreateSPIRVIntegerType(BitWidth, MIB);
}

static bool deduceAndAssignTypeForGUnmerge(MachineInstr *I, MachineFunction &MF,
                                           SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register SrcReg = I->getOperand(I->getNumOperands() - 1).getReg();
  SPIRVType *ScalarType = nullptr;
  if (SPIRVType *DefType = GR->getSPIRVTypeForVReg(SrcReg)) {
    assert(DefType->getOpcode() == SPIRV::OpTypeVector);
    ScalarType = GR->getSPIRVTypeForVReg(DefType->getOperand(1).getReg());
  }

  if (!ScalarType) {
    // If we could not deduce the type from the source, try to deduce it from
    // the uses of the results.
    for (unsigned i = 0; i < I->getNumDefs() && !ScalarType; ++i) {
      for (const auto &Use :
           MRI.use_nodbg_instructions(I->getOperand(i).getReg())) {
        assert(Use.getOpcode() == TargetOpcode::G_BUILD_VECTOR &&
               "Expected use of G_UNMERGE_VALUES to be a G_BUILD_VECTOR");
        if (auto *VecType =
                GR->getSPIRVTypeForVReg(Use.getOperand(0).getReg())) {
          ScalarType = GR->getScalarOrVectorComponentType(VecType);
          break;
        }
      }
    }
  }

  if (!ScalarType)
    return false;

  for (unsigned i = 0; i < I->getNumDefs(); ++i) {
    Register DefReg = I->getOperand(i).getReg();
    if (GR->getSPIRVTypeForVReg(DefReg))
      continue;

    LLT DefLLT = MRI.getType(DefReg);
    SPIRVType *ResType =
        DefLLT.isVector()
            ? GR->getOrCreateSPIRVVectorType(
                  ScalarType, DefLLT.getNumElements(), *I,
                  *MF.getSubtarget<SPIRVSubtarget>().getInstrInfo())
            : ScalarType;
    setRegClassType(DefReg, ResType, GR, &MRI, MF);
  }
  return true;
}

static SPIRVType *deduceTypeForGExtractVectorElt(MachineInstr *I,
                                                 MachineFunction &MF,
                                                 SPIRVGlobalRegistry *GR,
                                                 Register ResVReg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register VecReg = I->getOperand(1).getReg();
  if (SPIRVType *VecType = GR->getSPIRVTypeForVReg(VecReg)) {
    assert(VecType->getOpcode() == SPIRV::OpTypeVector);
    return GR->getScalarOrVectorComponentType(VecType);
  }

  // If not handled yet, then check if it is used in a G_BUILD_VECTOR.
  // If so get the type from there.
  for (const auto &Use : MRI.use_nodbg_instructions(ResVReg)) {
    if (Use.getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
      Register BuildVecResReg = Use.getOperand(0).getReg();
      if (SPIRVType *BuildVecType = GR->getSPIRVTypeForVReg(BuildVecResReg))
        return GR->getScalarOrVectorComponentType(BuildVecType);
    }
  }
  return nullptr;
}

static SPIRVType *deduceTypeForGBuildVector(MachineInstr *I,
                                            MachineFunction &MF,
                                            SPIRVGlobalRegistry *GR,
                                            MachineIRBuilder &MIB,
                                            Register ResVReg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  LLVM_DEBUG(dbgs() << "deduceTypeForGBuildVector: Processing " << *I << "\n");
  // First check if any of the operands have a type.
  for (unsigned i = 1; i < I->getNumOperands(); ++i) {
    if (SPIRVType *OpType =
            GR->getSPIRVTypeForVReg(I->getOperand(i).getReg())) {
      const LLT &ResLLT = MRI.getType(ResVReg);
      LLVM_DEBUG(dbgs() << "deduceTypeForGBuildVector: Found operand type "
                        << *OpType << ", returning vector type\n");
      return GR->getOrCreateSPIRVVectorType(OpType, ResLLT.getNumElements(),
                                            MIB, false);
    }
  }
  // If that did not work, then check the uses.
  for (const auto &Use : MRI.use_nodbg_instructions(ResVReg)) {
    if (Use.getOpcode() == TargetOpcode::G_EXTRACT_VECTOR_ELT) {
      Register ExtractResReg = Use.getOperand(0).getReg();
      if (SPIRVType *ScalarType = GR->getSPIRVTypeForVReg(ExtractResReg)) {
        const LLT &ResLLT = MRI.getType(ResVReg);
        LLVM_DEBUG(dbgs() << "deduceTypeForGBuildVector: Found use type "
                          << *ScalarType << ", returning vector type\n");
        return GR->getOrCreateSPIRVVectorType(
            ScalarType, ResLLT.getNumElements(), MIB, false);
      }
    }
  }
  LLVM_DEBUG(dbgs() << "deduceTypeForGBuildVector: Could not deduce type\n");
  return nullptr;
}

static SPIRVType *deduceTypeForGShuffleVector(MachineInstr *I,
                                              MachineFunction &MF,
                                              SPIRVGlobalRegistry *GR,
                                              MachineIRBuilder &MIB,
                                              Register ResVReg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const LLT &ResLLT = MRI.getType(ResVReg);
  assert(ResLLT.isVector() && "G_SHUFFLE_VECTOR result must be a vector");

  // The result element type should be the same as the input vector element
  // types.
  for (unsigned i = 1; i <= 2; ++i) {
    Register VReg = I->getOperand(i).getReg();
    if (auto *VType = GR->getSPIRVTypeForVReg(VReg)) {
      if (auto *ScalarType = GR->getScalarOrVectorComponentType(VType))
        return GR->getOrCreateSPIRVVectorType(
            ScalarType, ResLLT.getNumElements(), MIB, false);
    }
  }
  return nullptr;
}

static SPIRVType *deduceTypeForGImplicitDef(MachineInstr *I,
                                            MachineFunction &MF,
                                            SPIRVGlobalRegistry *GR,
                                            Register ResVReg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (const MachineInstr &Use : MRI.use_nodbg_instructions(ResVReg)) {
    SPIRVType *ScalarType = nullptr;
    switch (Use.getOpcode()) {
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_UNMERGE_VALUES:
      // It's possible that the use instruction has not been processed yet.
      // We should look at the operands of the use to determine the type.
      for (unsigned i = 1; i < Use.getNumOperands(); ++i) {
        if (SPIRVType *OpType =
                GR->getSPIRVTypeForVReg(Use.getOperand(i).getReg()))
          ScalarType = GR->getScalarOrVectorComponentType(OpType);
      }
      break;
    case TargetOpcode::G_SHUFFLE_VECTOR:
      // For G_SHUFFLE_VECTOR, only look at the vector input operands.
      if (auto *Type = GR->getSPIRVTypeForVReg(Use.getOperand(1).getReg()))
        ScalarType = GR->getScalarOrVectorComponentType(Type);
      if (auto *Type = GR->getSPIRVTypeForVReg(Use.getOperand(2).getReg()))
        ScalarType = GR->getScalarOrVectorComponentType(Type);
      break;
    }
    if (ScalarType) {
      const LLT &ResLLT = MRI.getType(ResVReg);
      if (!ResLLT.isVector())
        return ScalarType;
      return GR->getOrCreateSPIRVVectorType(
          ScalarType, ResLLT.getNumElements(), *I,
          *MF.getSubtarget<SPIRVSubtarget>().getInstrInfo());
    }
  }
  return nullptr;
}

static SPIRVType *deduceTypeForGIntrinsic(MachineInstr *I, MachineFunction &MF,
                                          SPIRVGlobalRegistry *GR,
                                          MachineIRBuilder &MIB,
                                          Register ResVReg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  if (!isSpvIntrinsic(*I, Intrinsic::spv_bitcast))
    return nullptr;

  for (const auto &Use : MRI.use_nodbg_instructions(ResVReg)) {
    const unsigned UseOpc = Use.getOpcode();
    assert(UseOpc == TargetOpcode::G_EXTRACT_VECTOR_ELT ||
           UseOpc == TargetOpcode::G_SHUFFLE_VECTOR ||
           UseOpc == TargetOpcode::G_BUILD_VECTOR ||
           UseOpc == TargetOpcode::G_UNMERGE_VALUES);
    Register UseResultReg = Use.getOperand(0).getReg();
    if (SPIRVType *UseResType = GR->getSPIRVTypeForVReg(UseResultReg)) {
      SPIRVType *ScalarType = GR->getScalarOrVectorComponentType(UseResType);
      const LLT &BitcastLLT = MRI.getType(ResVReg);
      if (BitcastLLT.isVector())
        return GR->getOrCreateSPIRVVectorType(
            ScalarType, BitcastLLT.getNumElements(), MIB, false);
      return ScalarType;
    }
  }
  return nullptr;
}

static SPIRVType *deduceTypeForGAnyExt(MachineInstr *I, MachineFunction &MF,
                                       SPIRVGlobalRegistry *GR,
                                       MachineIRBuilder &MIB,
                                       Register ResVReg) {
  // The result type of G_ANYEXT cannot be inferred from its operand.
  // We use the result register's LLT to determine the correct integer type.
  const LLT &ResLLT = MIB.getMRI()->getType(ResVReg);
  if (!ResLLT.isScalar())
    return nullptr;
  return GR->getOrCreateSPIRVIntegerType(ResLLT.getSizeInBits(), MIB);
}

static SPIRVType *deduceTypeForDefault(MachineInstr *I, MachineFunction &MF,
                                       SPIRVGlobalRegistry *GR) {
  if (I->getNumDefs() != 1 || I->getNumOperands() <= 1 ||
      !I->getOperand(1).isReg())
    return nullptr;

  SPIRVType *OpType = GR->getSPIRVTypeForVReg(I->getOperand(1).getReg());
  if (!OpType)
    return nullptr;
  return OpType;
}

static bool deduceAndAssignSpirvType(MachineInstr *I, MachineFunction &MF,
                                     SPIRVGlobalRegistry *GR,
                                     MachineIRBuilder &MIB) {
  LLVM_DEBUG(dbgs() << "Processing instruction: " << *I);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const unsigned Opcode = I->getOpcode();
  Register ResVReg = I->getOperand(0).getReg();
  SPIRVType *ResType = nullptr;

  switch (Opcode) {
  case TargetOpcode::G_CONSTANT: {
    ResType = deduceTypeForGConstant(I, MF, GR, MIB, ResVReg);
    break;
  }
  case TargetOpcode::G_UNMERGE_VALUES: {
    // This one is special as it defines multiple registers.
    if (deduceAndAssignTypeForGUnmerge(I, MF, GR))
      return true;
    break;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    ResType = deduceTypeForGExtractVectorElt(I, MF, GR, ResVReg);
    break;
  }
  case TargetOpcode::G_BUILD_VECTOR: {
    ResType = deduceTypeForGBuildVector(I, MF, GR, MIB, ResVReg);
    break;
  }
  case TargetOpcode::G_SHUFFLE_VECTOR: {
    ResType = deduceTypeForGShuffleVector(I, MF, GR, MIB, ResVReg);
    break;
  }
  case TargetOpcode::G_ANYEXT: {
    ResType = deduceTypeForGAnyExt(I, MF, GR, MIB, ResVReg);
    break;
  }
  case TargetOpcode::G_IMPLICIT_DEF: {
    ResType = deduceTypeForGImplicitDef(I, MF, GR, ResVReg);
    break;
  }
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS: {
    ResType = deduceTypeForGIntrinsic(I, MF, GR, MIB, ResVReg);
    break;
  }
  default:
    ResType = deduceTypeForDefault(I, MF, GR);
    break;
  }

  if (ResType) {
    LLVM_DEBUG(dbgs() << "Assigned type to " << *I << ": " << *ResType << "\n");
    GR->assignSPIRVTypeToVReg(ResType, ResVReg, MF);

    if (!MRI.getRegClassOrNull(ResVReg)) {
      LLVM_DEBUG(dbgs() << "Updating the register class.\n");
      setRegClassType(ResVReg, ResType, GR, &MRI, *GR->CurMF, true);
    }
    return true;
  }
  return false;
}

static bool requiresSpirvType(MachineInstr &I, SPIRVGlobalRegistry *GR,
                              MachineRegisterInfo &MRI) {
  LLVM_DEBUG(dbgs() << "Checking if instruction requires a SPIR-V type: "
                    << I;);
  if (I.getNumDefs() == 0) {
    LLVM_DEBUG(dbgs() << "Instruction does not have a definition.\n");
    return false;
  }
  if (!mayBeInserted(I.getOpcode())) {
    LLVM_DEBUG(dbgs() << "Instruction may not be inserted.\n");
    return false;
  }

  Register ResultRegister = I.defs().begin()->getReg();
  if (GR->getSPIRVTypeForVReg(ResultRegister)) {
    LLVM_DEBUG(dbgs() << "Instruction already has a SPIR-V type.\n");
    if (!MRI.getRegClassOrNull(ResultRegister)) {
      LLVM_DEBUG(dbgs() << "Updating the register class.\n");
      setRegClassType(ResultRegister, GR->getSPIRVTypeForVReg(ResultRegister),
                      GR, &MRI, *GR->CurMF, true);
    }
    return false;
  }

  return true;
}

static void registerSpirvTypeForNewInstructions(MachineFunction &MF,
                                                SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<MachineInstr *, 8> Worklist;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &I : MBB) {
      if (requiresSpirvType(I, GR, MRI)) {
        Worklist.push_back(&I);
      }
    }
  }

  if (Worklist.empty()) {
    LLVM_DEBUG(dbgs() << "Initial worklist is empty.\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "Initial worklist:\n";
             for (auto *I : Worklist) { I->dump(); });

  bool Changed = true;
  while (Changed) {
    Changed = false;
    SmallVector<MachineInstr *, 8> NextWorklist;

    for (MachineInstr *I : Worklist) {
      MachineIRBuilder MIB(*I);
      if (deduceAndAssignSpirvType(I, MF, GR, MIB)) {
        Changed = true;
      } else {
        NextWorklist.push_back(I);
      }
    }
    Worklist = NextWorklist;
    LLVM_DEBUG(dbgs() << "Worklist size: " << Worklist.size() << "\n");
  }

  if (!Worklist.empty()) {
    LLVM_DEBUG(dbgs() << "Remaining worklist:\n";
               for (auto *I : Worklist) { I->dump(); });
    for (auto *I : Worklist) {
      MachineIRBuilder MIB(*I);
      Register ResVReg = I->getOperand(0).getReg();
      const LLT &ResLLT = MRI.getType(ResVReg);
      SPIRVType *ResType = nullptr;
      if (ResLLT.isVector()) {
        SPIRVType *CompType = GR->getOrCreateSPIRVIntegerType(
            ResLLT.getElementType().getSizeInBits(), MIB);
        ResType = GR->getOrCreateSPIRVVectorType(
            CompType, ResLLT.getNumElements(), MIB, false);
      } else {
        ResType = GR->getOrCreateSPIRVIntegerType(ResLLT.getSizeInBits(), MIB);
      }
      setRegClassType(ResVReg, ResType, GR, &MRI, MF, true);
    }
  }
}

static void ensureAssignTypeForTypeFolding(MachineFunction &MF,
                                           SPIRVGlobalRegistry *GR) {
  LLVM_DEBUG(dbgs() << "Entering ensureAssignTypeForTypeFolding for function "
                    << MF.getName() << "\n");
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isTypeFoldingSupported(MI.getOpcode()))
        continue;
      if (MI.getNumOperands() == 1 || !MI.getOperand(1).isReg())
        continue;

      LLVM_DEBUG(dbgs() << "Processing instruction: " << MI);

      // Check uses of MI to see if it already has an use in SPIRV::ASSIGN_TYPE
      bool HasAssignType = false;
      Register ResultRegister = MI.defs().begin()->getReg();
      // All uses of Result register
      for (MachineInstr &UseInstr :
           MRI.use_nodbg_instructions(ResultRegister)) {
        if (UseInstr.getOpcode() == SPIRV::ASSIGN_TYPE) {
          HasAssignType = true;
          LLVM_DEBUG(dbgs() << "  Instruction already has an ASSIGN_TYPE use: "
                            << UseInstr);
          break;
        }
      }

      if (!HasAssignType) {
        Register ResultRegister = MI.defs().begin()->getReg();
        SPIRVType *ResultType = GR->getSPIRVTypeForVReg(ResultRegister);
        LLVM_DEBUG(
            dbgs() << "  Adding ASSIGN_TYPE for ResultRegister: "
                   << printReg(ResultRegister, MRI.getTargetRegisterInfo())
                   << " with type: " << *ResultType);
        MachineIRBuilder MIB(MI);
        insertAssignInstr(ResultRegister, nullptr, ResultType, GR, MIB, MRI);
      }
    }
  }
}

static void lowerExtractVectorElements(MachineFunction &MF) {
  SmallVector<MachineInstr *, 8> ExtractInstrs;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == TargetOpcode::G_EXTRACT_VECTOR_ELT) {
        ExtractInstrs.push_back(&MI);
      }
    }
  }

  for (MachineInstr *MI : ExtractInstrs) {
    MachineIRBuilder MIB(*MI);
    Register Dst = MI->getOperand(0).getReg();
    Register Vec = MI->getOperand(1).getReg();
    Register Idx = MI->getOperand(2).getReg();

    auto Intr = MIB.buildIntrinsic(Intrinsic::spv_extractelt, Dst, true, false);
    Intr.addUse(Vec);
    Intr.addUse(Idx);

    MI->eraseFromParent();
  }
}

// Do a preorder traversal of the CFG starting from the BB |Start|.
// point. Calls |op| on each basic block encountered during the traversal.
void visit(MachineFunction &MF, MachineBasicBlock &Start,
           std::function<void(MachineBasicBlock *)> op) {
  std::stack<MachineBasicBlock *> ToVisit;
  SmallPtrSet<MachineBasicBlock *, 8> Seen;

  ToVisit.push(&Start);
  Seen.insert(ToVisit.top());
  while (ToVisit.size() != 0) {
    MachineBasicBlock *MBB = ToVisit.top();
    ToVisit.pop();

    op(MBB);

    for (auto Succ : MBB->successors()) {
      if (Seen.contains(Succ))
        continue;
      ToVisit.push(Succ);
      Seen.insert(Succ);
    }
  }
}

// Do a preorder traversal of the CFG starting from the given function's entry
// point. Calls |op| on each basic block encountered during the traversal.
void visit(MachineFunction &MF, std::function<void(MachineBasicBlock *)> op) {
  visit(MF, *MF.begin(), std::move(op));
}

bool SPIRVPostLegalizer::runOnMachineFunction(MachineFunction &MF) {
  // Initialize the type registry.
  const SPIRVSubtarget &ST = MF.getSubtarget<SPIRVSubtarget>();
  SPIRVGlobalRegistry *GR = ST.getSPIRVGlobalRegistry();
  GR->setCurrentFunc(MF);
  registerSpirvTypeForNewInstructions(MF, GR);
  ensureAssignTypeForTypeFolding(MF, GR);
  lowerExtractVectorElements(MF);

  return true;
}

INITIALIZE_PASS(SPIRVPostLegalizer, DEBUG_TYPE, "SPIRV post legalizer", false,
                false)

char SPIRVPostLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPostLegalizerPass() {
  return new SPIRVPostLegalizer();
}
