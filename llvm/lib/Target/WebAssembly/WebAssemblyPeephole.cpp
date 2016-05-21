//===-- WebAssemblyPeephole.cpp - WebAssembly Peephole Optimiztions -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Late peephole optimizations for WebAssembly.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-peephole"

static cl::opt<bool> DisableWebAssemblyFallthroughReturnOpt(
    "disable-wasm-fallthrough-return-opt", cl::Hidden,
    cl::desc("WebAssembly: Disable fallthrough-return optimizations."),
    cl::init(false));

namespace {
class WebAssemblyPeephole final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly late peephole optimizer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID;
  WebAssemblyPeephole() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyPeephole::ID = 0;
FunctionPass *llvm::createWebAssemblyPeephole() {
  return new WebAssemblyPeephole();
}

/// If desirable, rewrite NewReg to a drop register.
static bool MaybeRewriteToDrop(unsigned OldReg, unsigned NewReg,
                               MachineOperand &MO, WebAssemblyFunctionInfo &MFI,
                               MachineRegisterInfo &MRI) {
  bool Changed = false;
  if (OldReg == NewReg) {
    Changed = true;
    unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(OldReg));
    MO.setReg(NewReg);
    MO.setIsDead();
    MFI.stackifyVReg(NewReg);
  }
  return Changed;
}

static bool MaybeRewriteToFallthrough(MachineInstr &MI, MachineBasicBlock &MBB,
                                      const MachineFunction &MF,
                                      WebAssemblyFunctionInfo &MFI,
                                      MachineRegisterInfo &MRI,
                                      const WebAssemblyInstrInfo &TII,
                                      unsigned FallthroughOpc,
                                      unsigned CopyLocalOpc) {
  if (DisableWebAssemblyFallthroughReturnOpt)
    return false;
  if (&MBB != &MF.back())
    return false;
  if (&MI != &MBB.back())
    return false;

  // If the operand isn't stackified, insert a COPY_LOCAL to read the operand
  // and stackify it.
  MachineOperand &MO = MI.getOperand(0);
  unsigned Reg = MO.getReg();
  if (!MFI.isVRegStackified(Reg)) {
    unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(Reg));
    BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(CopyLocalOpc), NewReg)
        .addReg(Reg);
    MO.setReg(NewReg);
    MFI.stackifyVReg(NewReg);
  }

  // Rewrite the return.
  MI.setDesc(TII.get(FallthroughOpc));
  return true;
}

bool WebAssemblyPeephole::runOnMachineFunction(MachineFunction &MF) {
  DEBUG({
    dbgs() << "********** Peephole **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const WebAssemblyTargetLowering &TLI =
      *MF.getSubtarget<WebAssemblySubtarget>().getTargetLowering();
  auto &LibInfo = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  bool Changed = false;

  for (auto &MBB : MF)
    for (auto &MI : MBB)
      switch (MI.getOpcode()) {
      default:
        break;
      case WebAssembly::STORE8_I32:
      case WebAssembly::STORE16_I32:
      case WebAssembly::STORE8_I64:
      case WebAssembly::STORE16_I64:
      case WebAssembly::STORE32_I64:
      case WebAssembly::STORE_F32:
      case WebAssembly::STORE_F64:
      case WebAssembly::STORE_I32:
      case WebAssembly::STORE_I64: {
        // Store instructions return their value operand. If we ended up using
        // the same register for both, replace it with a dead def so that it
        // can use $drop instead.
        MachineOperand &MO = MI.getOperand(0);
        unsigned OldReg = MO.getReg();
        unsigned NewReg =
            MI.getOperand(WebAssembly::StoreValueOperandNo).getReg();
        Changed |= MaybeRewriteToDrop(OldReg, NewReg, MO, MFI, MRI);
        break;
      }
      case WebAssembly::CALL_I32:
      case WebAssembly::CALL_I64: {
        MachineOperand &Op1 = MI.getOperand(1);
        if (Op1.isSymbol()) {
          StringRef Name(Op1.getSymbolName());
          if (Name == TLI.getLibcallName(RTLIB::MEMCPY) ||
              Name == TLI.getLibcallName(RTLIB::MEMMOVE) ||
              Name == TLI.getLibcallName(RTLIB::MEMSET)) {
            LibFunc::Func Func;
            if (LibInfo.getLibFunc(Name, Func)) {
              const auto &Op2 = MI.getOperand(2);
              if (!Op2.isReg())
                report_fatal_error("Peephole: call to builtin function with "
                                   "wrong signature, not consuming reg");
              MachineOperand &MO = MI.getOperand(0);
              unsigned OldReg = MO.getReg();
              unsigned NewReg = Op2.getReg();

              if (MRI.getRegClass(NewReg) != MRI.getRegClass(OldReg))
                report_fatal_error("Peephole: call to builtin function with "
                                   "wrong signature, from/to mismatch");
              Changed |= MaybeRewriteToDrop(OldReg, NewReg, MO, MFI, MRI);
            }
          }
        }
        break;
      }
      // Optimize away an explicit void return at the end of the function.
      case WebAssembly::RETURN_I32:
        Changed |= MaybeRewriteToFallthrough(
            MI, MBB, MF, MFI, MRI, TII, WebAssembly::FALLTHROUGH_RETURN_I32,
            WebAssembly::COPY_LOCAL_I32);
        break;
      case WebAssembly::RETURN_I64:
        Changed |= MaybeRewriteToFallthrough(
            MI, MBB, MF, MFI, MRI, TII, WebAssembly::FALLTHROUGH_RETURN_I64,
            WebAssembly::COPY_LOCAL_I64);
        break;
      case WebAssembly::RETURN_F32:
        Changed |= MaybeRewriteToFallthrough(
            MI, MBB, MF, MFI, MRI, TII, WebAssembly::FALLTHROUGH_RETURN_F32,
            WebAssembly::COPY_LOCAL_F32);
        break;
      case WebAssembly::RETURN_F64:
        Changed |= MaybeRewriteToFallthrough(
            MI, MBB, MF, MFI, MRI, TII, WebAssembly::FALLTHROUGH_RETURN_F64,
            WebAssembly::COPY_LOCAL_F64);
        break;
      case WebAssembly::RETURN_VOID:
        if (!DisableWebAssemblyFallthroughReturnOpt &&
            &MBB == &MF.back() && &MI == &MBB.back())
          MI.setDesc(TII.get(WebAssembly::FALLTHROUGH_RETURN_VOID));
        break;
      }

  return Changed;
}
