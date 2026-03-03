//===-- AArch64CodeLayoutOpt.cpp - Code Layout Optimizations --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs after instruction scheduling and employs code layout
// optimizations for certain patterns.
//
// Option -aarch64-code-layout-opt is a bitmask enable for instruction pairs of:
//   Bit 0 (0x1): Enable FCMP-FCSEL code layout optimization
//   Bit 1 (0x2): Enable CMP/CMN-CSEL code layout optimization
//
// The initial implementation induces function alignment to help optimize
// code layout for the detected patterns.
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-code-layout-opt"
#define AARCH64_CODE_LAYOUT_OPT_NAME "AArch64 Code Layout Optimization"

// Bitmask option for code alignment optimization:
//   Bit 0 (0x1): Enable FCMP-FCSEL code layout optimization (requires
//                hasFuseFCmpFCSel)
//   Bit 1 (0x2): Enable CMP-CSEL code layout optimization,
//                32-bit only (requires hasFuseCmpCSel)
static cl::opt<unsigned> EnableCodeAlignment(
    "aarch64-code-layout-opt", cl::Hidden,
    cl::desc("Enable code alignment optimization for instruction pairs "
             "(bitmask: bit 0 = FCMP-FCSEL, bit 1 = CMP-CSEL)"),
    cl::init(0));

static cl::opt<unsigned> FunctionAlignBytes(
    "aarch64-code-layout-opt-align-functions", cl::Hidden,
    cl::desc("Function alignment in bytes for code layout optimization "
             "(must be a power of 2)"),
    cl::init(64), cl::callback([](const unsigned &Val) {
      if (!isPowerOf2_32(Val))
        report_fatal_error(
            "aarch64-code-layout-opt-align must be a power of 2");
    }));

STATISTIC(NumFunctionsAligned,
          "Number of functions with aligned (to 64-bytes by default)");
STATISTIC(NumFcmpFcselPairsDetected,
          "Number of FCMP-FCSEL pairs detected for alignment");
STATISTIC(NumCmpCselPairsDetected,
          "Number of CMP/CMN-CSEL pairs detected for alignment");

namespace {

class AArch64CodeLayoutOpt : public MachineFunctionPass {
public:
  static char ID;
  AArch64CodeLayoutOpt() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return AARCH64_CODE_LAYOUT_OPT_NAME;
  }

private:
  const AArch64InstrInfo *TII = nullptr;

  // Returns true if MBB contains at least one layout-sensitive pattern.
  bool detectLayoutSensitivePattern(MachineBasicBlock *MBB);

  bool optimizeForCodeAlignment(MachineFunction &MF);
};

} // end anonymous namespace

char AArch64CodeLayoutOpt::ID = 0;

INITIALIZE_PASS(AArch64CodeLayoutOpt, "aarch64-code-layout-opt",
                AARCH64_CODE_LAYOUT_OPT_NAME, false, false)

void AArch64CodeLayoutOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createAArch64CodeLayoutOptPass() {
  return new AArch64CodeLayoutOpt();
}

bool AArch64CodeLayoutOpt::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableCodeAlignment)
    return false;

  const auto *Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();

  const unsigned Mask = EnableCodeAlignment;
  if (!((Mask & 0x1) && Subtarget->hasFuseFCmpFCSel()) &&
      !((Mask & 0x2) && Subtarget->hasFuseCmpCSel()))
    return false;

  return optimizeForCodeAlignment(MF);
}

// Returns true if MBB contains at least one layout-sensitive pair.
// A pair is: a qualifying lead instruction immediately followed by its
// consumer (FCMP→FCSEL or CMP/CMN→CSEL), with no intervening instructions.
bool AArch64CodeLayoutOpt::detectLayoutSensitivePattern(
    MachineBasicBlock *MBB) {
  MachineInstr *PendingFCMPInstr = nullptr;
  MachineInstr *PendingCMPInstr = nullptr;

  for (auto &MI : instructionsWithoutDebug(MBB->begin(), MBB->end())) {
    if (MI.isMetaInstruction())
      continue;

    unsigned Opc = MI.getOpcode();

    // --- FCMP-FCSEL detection (bit 0) ---
    if (EnableCodeAlignment & 0x1) {
      switch (Opc) {
      case AArch64::FCMPSrr:
      case AArch64::FCMPDrr:
      case AArch64::FCMPESrr:
      case AArch64::FCMPEDrr:
      case AArch64::FCMPHrr:
      case AArch64::FCMPEHrr:
        PendingFCMPInstr = &MI;
        break;
      case AArch64::FCSELSrrr:
      case AArch64::FCSELDrrr:
      case AArch64::FCSELHrrr:
        if (PendingFCMPInstr) {
          ++NumFcmpFcselPairsDetected;
          return true;
        }
        PendingFCMPInstr = nullptr;
        break;
      default:
        PendingFCMPInstr = nullptr;
        break;
      }
    }

    // --- CMP/CMN-CSEL detection (bit 1) ---
    // CMP is encoded as SUBS with WZR destination (32-bit only).
    // CMN is encoded as ADDS with WZR destination (32-bit only).
    // Only simple variants (no shifted/extended reg) qualify.
    if (EnableCodeAlignment & 0x2) {
      bool IsCMP = false;
      switch (Opc) {
      case AArch64::SUBSWrr:
      case AArch64::ADDSWrr:
        IsCMP = MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr);
        break;
      case AArch64::SUBSWri:
      case AArch64::ADDSWri:
        // Only CMP/CMN #imm (no LSL #12 shift) with small immediates (<=15)
        IsCMP = MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
                MI.getOperand(3).getImm() == 0 &&
                MI.getOperand(2).getImm() <= 15;
        break;
      case AArch64::SUBSWrs:
      case AArch64::ADDSWrs:
        IsCMP = MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
                !AArch64InstrInfo::hasShiftedReg(MI);
        break;
      case AArch64::SUBSWrx:
        IsCMP = MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
                !AArch64InstrInfo::hasExtendedReg(MI);
        break;
      case AArch64::CSELWr:
        if (PendingCMPInstr) {
          ++NumCmpCselPairsDetected;
          return true;
        }
        PendingCMPInstr = nullptr;
        break;
      default:
        break;
      }

      if (IsCMP)
        PendingCMPInstr = &MI;
      else if (Opc != AArch64::CSELWr)
        PendingCMPInstr = nullptr;
    }
  }

  return false;
}

bool AArch64CodeLayoutOpt::optimizeForCodeAlignment(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << DEBUG_TYPE ": optimizeForCodeAlignment: " << MF.getName()
                    << "\n");

  for (auto &MBB : MF) {
    if (!detectLayoutSensitivePattern(&MBB))
      continue;

    if (MF.getAlignment() >= Align(FunctionAlignBytes)) {
      LLVM_DEBUG(dbgs() << DEBUG_TYPE ": Function " << MF.getName()
                        << " already has sufficient alignment\n");
      return false;
    }

    MF.setAlignment(Align(FunctionAlignBytes));
    ++NumFunctionsAligned;
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": Set " << FunctionAlignBytes
                      << "-byte alignment for function " << MF.getName()
                      << "\n");
    return true;
  }

  return false;
}
