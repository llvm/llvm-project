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
#include "llvm/IR/Attributes.h"
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
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMAX:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMAXIMUM:
    return true;
  default:
    return isTypeFoldingSupported(Opcode);
  }
}

static void processNewInstrs(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                             MachineIRBuilder MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &I : MBB) {
      const unsigned Opcode = I.getOpcode();
      if (Opcode == TargetOpcode::G_UNMERGE_VALUES) {
        unsigned ArgI = I.getNumOperands() - 1;
        Register SrcReg = I.getOperand(ArgI).isReg()
                              ? I.getOperand(ArgI).getReg()
                              : Register(0);
        SPIRVType *DefType =
            SrcReg.isValid() ? GR->getSPIRVTypeForVReg(SrcReg) : nullptr;
        if (!DefType || DefType->getOpcode() != SPIRV::OpTypeVector)
          report_fatal_error(
              "cannot select G_UNMERGE_VALUES with a non-vector argument");
        SPIRVType *ScalarType =
            GR->getSPIRVTypeForVReg(DefType->getOperand(1).getReg());
        for (unsigned i = 0; i < I.getNumDefs(); ++i) {
          Register ResVReg = I.getOperand(i).getReg();
          SPIRVType *ResType = GR->getSPIRVTypeForVReg(ResVReg);
          if (!ResType) {
            // There was no "assign type" actions, let's fix this now
            ResType = ScalarType;
            setRegClassType(ResVReg, ResType, GR, &MRI, *GR->CurMF, true);
          }
        }
      } else if (mayBeInserted(Opcode) && I.getNumDefs() == 1 &&
                 I.getNumOperands() > 1 && I.getOperand(1).isReg()) {
        // Legalizer may have added a new instructions and introduced new
        // registers, we must decorate them as if they were introduced in a
        // non-automatic way
        Register ResVReg = I.getOperand(0).getReg();
        // Check if the register defined by the instruction is newly generated
        // or already processed
        // Check if we have type defined for operands of the new instruction
        bool IsKnownReg = MRI.getRegClassOrNull(ResVReg);
        SPIRVType *ResVType = GR->getSPIRVTypeForVReg(
            IsKnownReg ? ResVReg : I.getOperand(1).getReg());
        if (!ResVType)
          continue;
        // Set type & class
        if (!IsKnownReg)
          setRegClassType(ResVReg, ResVType, GR, &MRI, *GR->CurMF, true);
        // If this is a simple operation that is to be reduced by TableGen
        // definition we must apply some of pre-legalizer rules here
        if (isTypeFoldingSupported(Opcode)) {
          processInstr(I, MIB, MRI, GR, GR->getSPIRVTypeForVReg(ResVReg));
          if (IsKnownReg && MRI.hasOneUse(ResVReg)) {
            MachineInstr &UseMI = *MRI.use_instr_begin(ResVReg);
            if (UseMI.getOpcode() == SPIRV::ASSIGN_TYPE)
              continue;
          }
          insertAssignInstr(ResVReg, nullptr, ResVType, GR, MIB, MRI);
        }
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
  MachineIRBuilder MIB(MF);

  processNewInstrs(MF, GR, MIB);

  return true;
}

INITIALIZE_PASS(SPIRVPostLegalizer, DEBUG_TYPE, "SPIRV post legalizer", false,
                false)

char SPIRVPostLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPostLegalizerPass() {
  return new SPIRVPostLegalizer();
}
