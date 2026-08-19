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
// Option -aarch64-code-layout-opt-enable selects instruction pairs to optimize:
//   cmp-csel:   Enable CMP/CMN-CSEL code layout optimization
//   fcmp-fcsel: Enable FCMP-FCSEL code layout optimization
//
// The initial implementation induces function alignment when a supported
// pattern is detected, and possibly instruction-alignment when a pair would
// straddle cache-lines.
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-code-layout-opt"
#define DBG(...) LLVM_DEBUG(dbgs() << DEBUG_TYPE ": " << __VA_ARGS__)
#define AARCH64_CODE_LAYOUT_OPT_NAME "AArch64 Code Layout Optimization"

enum CodeLayoutOpt {
  CmpCsel,   // Align CMP/CMN-CSEL pairs
  FcmpFcsel, // Align FCMP-FCSEL pairs
};

static cl::bits<CodeLayoutOpt> EnableCodeAlignment(
    "aarch64-code-layout-opt-enable", cl::Hidden, cl::CommaSeparated,
    cl::desc("Enable code alignment optimization for instruction pairs"),
    cl::values(
        clEnumValN(CmpCsel, "cmp-csel", "CMP/CMN-CSEL pair alignment (32-bit)"),
        clEnumValN(FcmpFcsel, "fcmp-fcsel", "FCMP-FCSEL pair alignment")));

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
STATISTIC(NumCmpCselPairsDetected,
          "Number of CMP/CMN-CSEL pairs detected for alignment");
STATISTIC(NumFcmpFcselPairsDetected,
          "Number of FCMP-FCSEL pairs detected for alignment");

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

  /// Align each fusible CMP/CMN-CSEL or FCMP-FCSEL pair in \p MBB by emitting
  /// .p2align before the lead instruction (splitting the block if needed).
  /// \returns true iff at least one pair was found and aligned.
  bool alignLayoutSensitivePatterns(MachineBasicBlock *MBB);

  /// Emit .p2align before MI. Splits the block if MI is not at its start.
  void emitP2Align(MachineInstr &MI, Align DesiredAlign,
                   unsigned MaxSkipBytes = 4);

  bool optimizeForCodeLayout(MachineFunction &MF);
};

} // end anonymous namespace

char AArch64CodeLayoutOpt::ID = 0;

INITIALIZE_PASS(AArch64CodeLayoutOpt, "aarch64-code-layout-opt",
                AARCH64_CODE_LAYOUT_OPT_NAME, false, false)

void AArch64CodeLayoutOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createAArch64CodeLayoutOptPass() {
  return new AArch64CodeLayoutOpt();
}

/// \returns true iff Opc is a floating-point comparison (FCMP/FCMPE).
static bool isFloatingPointCompare(unsigned Opc) {
  switch (Opc) {
  case AArch64::FCMPSrr:
  case AArch64::FCMPDrr:
  case AArch64::FCMPESrr:
  case AArch64::FCMPEDrr:
  case AArch64::FCMPHrr:
  case AArch64::FCMPEHrr:
    return true;
  default:
    return false;
  }
}

/// \returns true iff Opc is a floating-point conditional select (FCSEL).
static bool isFloatingPointConditionalSelect(unsigned Opc) {
  switch (Opc) {
  case AArch64::FCSELSrrr:
  case AArch64::FCSELDrrr:
  case AArch64::FCSELHrrr:
    return true;
  default:
    return false;
  }
}

/// \returns true if MI is a qualifying 32-bit CMP or CMN instruction.
/// CMP is encoded as SUBS with WZR destination, CMN as ADDS with WZR.
/// Only simple variants (no shifted/extended reg) qualify, and immediate
/// variants require no LSL shift and small immediates (<=15).
static bool isQualifyingIntCompare(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AArch64::SUBSWrr:
  case AArch64::ADDSWrr:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr);
  case AArch64::SUBSWri:
  case AArch64::ADDSWri:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
           MI.getOperand(3).getImm() == 0 && MI.getOperand(2).getImm() <= 15;
  case AArch64::SUBSWrs:
  case AArch64::ADDSWrs:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
           !AArch64InstrInfo::hasShiftedReg(MI);
  case AArch64::SUBSWrx:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
           !AArch64InstrInfo::hasExtendedReg(MI);
  default:
    return false;
  }
}

bool AArch64CodeLayoutOpt::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  // hasOptSize() returns true for both -Os and -Oz.
  if (F.hasOptSize())
    return false;

  const auto *Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();

  // Default: enable when the subtarget opts in via FeatureAlignCmpCSelPairs.
  if (!EnableCodeAlignment.getBits() && Subtarget->hasAlignCmpCSelPairs()) {
    if (Subtarget->hasFuseCmpCSel())
      EnableCodeAlignment.addValue(CmpCsel);
    if (Subtarget->hasFuseFCmpFCSel())
      EnableCodeAlignment.addValue(FcmpFcsel);
  }

  if (!(EnableCodeAlignment.isSet(CmpCsel) && Subtarget->hasFuseCmpCSel()) &&
      !(EnableCodeAlignment.isSet(FcmpFcsel) && Subtarget->hasFuseFCmpFCSel()))
    return false;

  return optimizeForCodeLayout(MF);
}

void AArch64CodeLayoutOpt::emitP2Align(MachineInstr &MI, Align DesiredAlign,
                                       unsigned MaxSkipBytes) {
  MachineBasicBlock *MBB = MI.getParent();

  auto FirstReal =
      skipDebugInstructionsForward(MBB->instr_begin(), MBB->instr_end());
  if (&*FirstReal != &MI) {
    auto PrevIt = prev_nodbg(MI.getIterator(), MBB->instr_begin());
    MBB = MBB->splitAt(*PrevIt, /*UpdateLiveIns=*/true);
  }

  MBB->setAlignment(DesiredAlign);
  MBB->setMaxBytesForAlignment(MaxSkipBytes);
}

// Align each fusible CMP/CMN-CSEL or FCMP-FCSEL pair in MBB by emitting
// .p2align before the lead instruction (splitting the block if needed).
// A pair is: a qualifying lead instruction immediately followed by its
// consumer (CMP/CMN→CSEL or FCMP→FCSEL), with no intervening instructions.
// Returns true iff at least one pair was found and aligned.
bool AArch64CodeLayoutOpt::alignLayoutSensitivePatterns(
    MachineBasicBlock *MBB) {
  auto End = MBB->instr_end();
  SmallVector<std::pair<MachineInstr *, bool>, 4> Pairs;

  for (auto &MI : instructionsWithoutDebug(MBB->begin(), MBB->end())) {
    auto NextIt =
        skipDebugInstructionsForward(std::next(MI.getIterator()), End);
    if (NextIt == End)
      break;

    // --- CMP/CMN-CSEL detection ---
    if (EnableCodeAlignment.isSet(CmpCsel) && isQualifyingIntCompare(MI) &&
        NextIt->getOpcode() == AArch64::CSELWr) {
      Pairs.push_back({&MI, true});
      continue;
    }

    // --- FCMP-FCSEL detection ---
    if (EnableCodeAlignment.isSet(FcmpFcsel) &&
        isFloatingPointCompare(MI.getOpcode()) &&
        isFloatingPointConditionalSelect(NextIt->getOpcode())) {
      Pairs.push_back({&MI, false});
      continue;
    }
  }

  for (auto &[MI, IsCmpCsel] : Pairs) {
    emitP2Align(*MI, Align(64));
    DBG(".p2align 6, , 4 before " << *MI);
    ++(IsCmpCsel ? NumCmpCselPairsDetected : NumFcmpFcselPairsDetected);
  }

  return !Pairs.empty();
}

bool AArch64CodeLayoutOpt::optimizeForCodeLayout(MachineFunction &MF) {
  DBG("optimizeForCodeLayout: " << MF.getName() << "\n");

  bool Changed = false;
  for (auto &MBB : MF)
    Changed |= alignLayoutSensitivePatterns(&MBB);

  if (!Changed)
    return false;

  if (MF.getAlignment() < Align(FunctionAlignBytes)) {
    MF.setAlignment(Align(FunctionAlignBytes));
    ++NumFunctionsAligned;
    DBG("Set " << FunctionAlignBytes << "-byte alignment for function "
               << MF.getName() << "\n");
  } else {
    DBG("Function " << MF.getName() << " already has sufficient alignment\n");
  }
  return true;
}
