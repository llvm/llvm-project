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
    return true;
  default:
    return isTypeFoldingSupported(Opcode);
  }
}

static bool processInstr(MachineInstr *I, MachineFunction &MF,
                         SPIRVGlobalRegistry *GR, MachineIRBuilder &MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const unsigned Opcode = I->getOpcode();
  Register ResVReg = I->getOperand(0).getReg();
  SPIRVType *ResType = nullptr;
  bool Handled = false;

  switch (Opcode) {
  case TargetOpcode::G_CONSTANT: {
    const LLT &Ty = MRI.getType(ResVReg);
    unsigned BitWidth = Ty.getScalarSizeInBits();
    ResType = GR->getOrCreateSPIRVIntegerType(BitWidth, MIB);
    Handled = true;
    break;
  }
  case TargetOpcode::G_UNMERGE_VALUES: {
    Register SrcReg = I->getOperand(I->getNumOperands() - 1).getReg();
    if (SPIRVType *DefType = GR->getSPIRVTypeForVReg(SrcReg)) {
      if (DefType->getOpcode() == SPIRV::OpTypeVector) {
        SPIRVType *ScalarType =
            GR->getSPIRVTypeForVReg(DefType->getOperand(1).getReg());
        for (unsigned i = 0; i < I->getNumDefs(); ++i) {
          Register DefReg = I->getOperand(i).getReg();
          if (!GR->getSPIRVTypeForVReg(DefReg)) {
            LLT DefLLT = MRI.getType(DefReg);
            SPIRVType *ResType;
            if (DefLLT.isVector()) {
              const SPIRVInstrInfo *TII =
                  MF.getSubtarget<SPIRVSubtarget>().getInstrInfo();
              ResType = GR->getOrCreateSPIRVVectorType(
                  ScalarType, DefLLT.getNumElements(), *I, *TII);
            } else {
              ResType = ScalarType;
            }
            setRegClassType(DefReg, ResType, GR, &MRI, MF);
          }
        }
        Handled = true;
      }
    }
    break;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    LLVM_DEBUG(dbgs() << "Processing G_EXTRACT_VECTOR_ELT: " << *I);
    Register VecReg = I->getOperand(1).getReg();
    if (SPIRVType *VecType = GR->getSPIRVTypeForVReg(VecReg)) {
      LLVM_DEBUG(dbgs() << "  Found vector type: " << *VecType << "\n");
      if (VecType->getOpcode() != SPIRV::OpTypeVector) {
        VecType->dump();
      }
      assert(VecType->getOpcode() == SPIRV::OpTypeVector);
      ResType = GR->getScalarOrVectorComponentType(VecType);
      Handled = true;
    } else {
      LLVM_DEBUG(dbgs() << "  Vector operand " << VecReg
                        << " has no type. Looking at uses of " << ResVReg
                        << ".\n");
      // If not handled yet, then check if it is used in a G_BUILD_VECTOR.
      // If so get the type from there.
      for (const auto &Use : MRI.use_nodbg_instructions(ResVReg)) {
        LLVM_DEBUG(dbgs() << "  Use: " << Use);
        if (Use.getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
          LLVM_DEBUG(dbgs() << "    Use is G_BUILD_VECTOR.\n");
          Register BuildVecResReg = Use.getOperand(0).getReg();
          if (SPIRVType *BuildVecType =
                  GR->getSPIRVTypeForVReg(BuildVecResReg)) {
            LLVM_DEBUG(dbgs() << "    Found G_BUILD_VECTOR result type: "
                              << *BuildVecType << "\n");
            ResType = GR->getScalarOrVectorComponentType(BuildVecType);
            Handled = true;
            break;
          } else {
            LLVM_DEBUG(dbgs() << "    G_BUILD_VECTOR result " << BuildVecResReg
                              << " has no type yet.\n");
          }
        }
      }
    }
    if (!Handled) {
      LLVM_DEBUG(
          dbgs() << "  Could not determine type for G_EXTRACT_VECTOR_ELT.\n");
    }
    break;
  }
  case TargetOpcode::G_BUILD_VECTOR: {
    // First check if any of the operands have a type.
    for (unsigned i = 1; i < I->getNumOperands(); ++i) {
      if (SPIRVType *OpType =
              GR->getSPIRVTypeForVReg(I->getOperand(i).getReg())) {
        const LLT &ResLLT = MRI.getType(ResVReg);
        ResType = GR->getOrCreateSPIRVVectorType(
            OpType, ResLLT.getNumElements(), MIB, false);
        Handled = true;
        break;
      }
    }
    if (Handled) {
      break;
    }
    // If that did not work, then check the uses.
    for (const auto &Use : MRI.use_nodbg_instructions(ResVReg)) {
      if (Use.getOpcode() == TargetOpcode::G_EXTRACT_VECTOR_ELT) {
        Register ExtractResReg = Use.getOperand(0).getReg();
        if (SPIRVType *ScalarType = GR->getSPIRVTypeForVReg(ExtractResReg)) {
          const LLT &ResLLT = MRI.getType(ResVReg);
          ResType = GR->getOrCreateSPIRVVectorType(
              ScalarType, ResLLT.getNumElements(), MIB, false);
          Handled = true;
          break;
        }
      }
    }
    break;
  }
  case TargetOpcode::G_IMPLICIT_DEF: {
    for (const auto &Use : MRI.use_nodbg_instructions(ResVReg)) {
      const unsigned UseOpc = Use.getOpcode();
      assert(UseOpc == TargetOpcode::G_BUILD_VECTOR ||
             UseOpc == TargetOpcode::G_SHUFFLE_VECTOR);
      // It's possible that the use instruction has not been processed yet.
      // We should look at the operands of the use to determine the type.
      for (unsigned i = 1; i < Use.getNumOperands(); ++i) {
        if (auto *Type = GR->getSPIRVTypeForVReg(Use.getOperand(i).getReg())) {
          ResType = Type;
          Handled = true;
          break;
        }
      }
      if (Handled) {
        break;
      }
    }
    break;
  }
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS: {
    if (!isSpvIntrinsic(*I, Intrinsic::spv_bitcast))
      break;

    for (const auto &Use : MRI.use_nodbg_instructions(ResVReg)) {
      const unsigned UseOpc = Use.getOpcode();
      assert(UseOpc == TargetOpcode::G_EXTRACT_VECTOR_ELT ||
             UseOpc == TargetOpcode::G_SHUFFLE_VECTOR);
      Register UseResultReg = Use.getOperand(0).getReg();
      if (SPIRVType *UseResType = GR->getSPIRVTypeForVReg(UseResultReg)) {
        SPIRVType *ScalarType = GR->getScalarOrVectorComponentType(UseResType);
        const LLT &BitcastLLT = MRI.getType(ResVReg);
        if (BitcastLLT.isVector()) {
          ResType = GR->getOrCreateSPIRVVectorType(
              ScalarType, BitcastLLT.getNumElements(), MIB, false);
        } else {
          ResType = ScalarType;
        }
        Handled = true;
        break;
      }
    }
    break;
  }
  default:
    if (I->getNumDefs() == 1 && I->getNumOperands() > 1 &&
        I->getOperand(1).isReg()) {
      if (SPIRVType *OpType =
              GR->getSPIRVTypeForVReg(I->getOperand(1).getReg())) {
        ResType = OpType;
        Handled = true;
      }
    }
    break;
  }

  if (Handled && ResType) {
    LLVM_DEBUG(dbgs() << "Assigned type to " << *I << ": " << *ResType << "\n");
    GR->assignSPIRVTypeToVReg(ResType, ResVReg, MF);
  }
  return Handled;
}

static void processNewInstrs(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                             MachineIRBuilder MIB) {
  SmallVector<MachineInstr *, 8> Worklist;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &I : MBB) {
      if (I.getNumDefs() > 0 &&
          !GR->getSPIRVTypeForVReg(I.getOperand(0).getReg()) &&
          mayBeInserted(I.getOpcode())) {
        Worklist.push_back(&I);
      }
    }
  }

  if (Worklist.empty()) {
    return;
  }

  LLVM_DEBUG(dbgs() << "Initial worklist:\n";
             for (auto *I : Worklist) { I->dump(); });

  bool Changed = true;
  while (Changed) {
    Changed = false;
    SmallVector<MachineInstr *, 8> NextWorklist;

    for (MachineInstr *I : Worklist) {
      if (processInstr(I, MF, GR, MIB)) {
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
    assert(Worklist.empty() && "Worklist is not empty");
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
  MachineIRBuilder MIB(MF);

  processNewInstrs(MF, GR, MIB);

  // TODO: Move this into is own function.
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
  return true;
}

INITIALIZE_PASS(SPIRVPostLegalizer, DEBUG_TYPE, "SPIRV post legalizer", false,
                false)

char SPIRVPostLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPostLegalizerPass() {
  return new SPIRVPostLegalizer();
}
