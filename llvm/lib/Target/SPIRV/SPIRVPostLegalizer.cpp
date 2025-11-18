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

static SPIRVType *deduceIntTypeFromResult(Register ResVReg,
                                          MachineIRBuilder &MIB,
                                          SPIRVGlobalRegistry *GR) {
  const LLT &Ty = MIB.getMRI()->getType(ResVReg);
  return GR->getOrCreateSPIRVIntegerType(Ty.getScalarSizeInBits(), MIB);
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

static SPIRVType *deduceTypeFromSingleOperand(MachineInstr *I,
                                              MachineIRBuilder &MIB,
                                              SPIRVGlobalRegistry *GR,
                                              unsigned OpIdx) {
  Register OpReg = I->getOperand(OpIdx).getReg();
  if (SPIRVType *OpType = GR->getSPIRVTypeForVReg(OpReg)) {
    if (SPIRVType *CompType = GR->getScalarOrVectorComponentType(OpType)) {
      Register ResVReg = I->getOperand(0).getReg();
      const LLT &ResLLT = MIB.getMRI()->getType(ResVReg);
      if (ResLLT.isVector())
        return GR->getOrCreateSPIRVVectorType(CompType, ResLLT.getNumElements(),
                                              MIB, false);
      return CompType;
    }
  }
  return nullptr;
}

static SPIRVType *deduceTypeFromOperandRange(MachineInstr *I,
                                             MachineIRBuilder &MIB,
                                             SPIRVGlobalRegistry *GR,
                                             unsigned StartOp, unsigned EndOp) {
  SPIRVType *ResType = nullptr;
  for (unsigned i = StartOp; i < EndOp; ++i) {
    if (SPIRVType *Type = deduceTypeFromSingleOperand(I, MIB, GR, i)) {
#ifdef EXPENSIVE_CHECKS
      assert(!ResType || Type == ResType && "Conflicting type from operands.");
      ResType = Type;
#else
      return Type;
#endif
    }
  }
  return ResType;
}

static SPIRVType *deduceTypeForResultRegister(MachineInstr *Use,
                                              Register UseRegister,
                                              SPIRVGlobalRegistry *GR,
                                              MachineIRBuilder &MIB) {
  for (const MachineOperand &MO : Use->defs()) {
    if (!MO.isReg())
      continue;
    if (SPIRVType *OpType = GR->getSPIRVTypeForVReg(MO.getReg())) {
      if (SPIRVType *CompType = GR->getScalarOrVectorComponentType(OpType)) {
        const LLT &ResLLT = MIB.getMRI()->getType(UseRegister);
        if (ResLLT.isVector())
          return GR->getOrCreateSPIRVVectorType(
              CompType, ResLLT.getNumElements(), MIB, false);
        return CompType;
      }
    }
  }
  return nullptr;
}

static SPIRVType *deduceTypeFromUses(Register Reg, MachineFunction &MF,
                                     SPIRVGlobalRegistry *GR,
                                     MachineIRBuilder &MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineInstr &Use : MRI.use_nodbg_instructions(Reg)) {
    SPIRVType *ResType = nullptr;
    switch (Use.getOpcode()) {
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_EXTRACT_VECTOR_ELT:
    case TargetOpcode::G_UNMERGE_VALUES:
      LLVM_DEBUG(dbgs() << "Looking at use " << Use << "\n");
      ResType = deduceTypeForResultRegister(&Use, Reg, GR, MIB);
      break;
    }
    if (ResType)
      return ResType;
  }
  return nullptr;
}

static SPIRVType *deduceResultTypeFromOperands(MachineInstr *I,
                                               SPIRVGlobalRegistry *GR,
                                               MachineIRBuilder &MIB) {
  Register ResVReg = I->getOperand(0).getReg();
  switch (I->getOpcode()) {
  case TargetOpcode::G_CONSTANT:
  case TargetOpcode::G_ANYEXT:
    return deduceIntTypeFromResult(ResVReg, MIB, GR);
  case TargetOpcode::G_BUILD_VECTOR:
    return deduceTypeFromOperandRange(I, MIB, GR, 1, I->getNumOperands());
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return deduceTypeFromOperandRange(I, MIB, GR, 1, 3);
  default:
    if (I->getNumDefs() == 1 && I->getNumOperands() > 1 &&
        I->getOperand(1).isReg())
      return deduceTypeFromSingleOperand(I, MIB, GR, 1);
    return nullptr;
  }
}

static bool deduceAndAssignSpirvType(MachineInstr *I, MachineFunction &MF,
                                     SPIRVGlobalRegistry *GR,
                                     MachineIRBuilder &MIB) {
  LLVM_DEBUG(dbgs() << "\nProcessing instruction: " << *I);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register ResVReg = I->getOperand(0).getReg();

  // G_UNMERGE_VALUES is handled separately because it has multiple definitions,
  // unlike the other instructions which have a single result register. The main
  // deduction logic is designed for the single-definition case.
  if (I->getOpcode() == TargetOpcode::G_UNMERGE_VALUES)
    return deduceAndAssignTypeForGUnmerge(I, MF, GR);

  LLVM_DEBUG(dbgs() << "Inferring type from operands\n");
  SPIRVType *ResType = deduceResultTypeFromOperands(I, GR, MIB);
  if (!ResType) {
    LLVM_DEBUG(dbgs() << "Inferring type from uses\n");
    ResType = deduceTypeFromUses(ResVReg, MF, GR, MIB);
  }

  if (!ResType)
    return false;

  LLVM_DEBUG(dbgs() << "Assigned type to " << *I << ": " << *ResType);
  GR->assignSPIRVTypeToVReg(ResType, ResVReg, MF);

  if (!MRI.getRegClassOrNull(ResVReg)) {
    LLVM_DEBUG(dbgs() << "Updating the register class.\n");
    setRegClassType(ResVReg, ResType, GR, &MRI, *GR->CurMF, true);
  }
  return true;
}

static bool requiresSpirvType(MachineInstr &I, SPIRVGlobalRegistry *GR,
                              MachineRegisterInfo &MRI) {
  LLVM_DEBUG(dbgs() << "Checking if instruction requires a SPIR-V type: "
                    << I;);
  if (I.getNumDefs() == 0) {
    LLVM_DEBUG(dbgs() << "Instruction does not have a definition.\n");
    return false;
  }

  if (!I.isPreISelOpcode()) {
    LLVM_DEBUG(dbgs() << "Instruction is not a generic instruction.\n");
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

  bool Changed;
  do {
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
    Worklist = std::move(NextWorklist);
    LLVM_DEBUG(dbgs() << "Worklist size: " << Worklist.size() << "\n");
  } while (Changed);

  if (Worklist.empty())
    return;

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
    LLVM_DEBUG(dbgs() << "Could not determine type for " << *I
                      << ", defaulting to " << *ResType << "\n");
    setRegClassType(ResVReg, ResType, GR, &MRI, MF, true);
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
  return true;
}

INITIALIZE_PASS(SPIRVPostLegalizer, DEBUG_TYPE, "SPIRV post legalizer", false,
                false)

char SPIRVPostLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPostLegalizerPass() {
  return new SPIRVPostLegalizer();
}
