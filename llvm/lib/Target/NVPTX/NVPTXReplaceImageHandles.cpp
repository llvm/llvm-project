//===-- NVPTXReplaceImageHandles.cpp - Replace image handles for Fermi ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// On Fermi, image handles are not supported. To work around this, we traverse
// the machine code and replace image handles with concrete symbols. For this
// to work reliably, inlining of all function call must be performed.
//
//===----------------------------------------------------------------------===//
#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTX.h"
#include "NVPTXMachineFunctionInfo.h"
#include "NVPTXSubtarget.h"
#include "NVPTXTargetMachine.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {
class NVPTXReplaceImageHandles {
  DenseSet<MachineInstr *> InstrsToRemove;

public:
  bool run(MachineFunction &MF);

private:
  bool processInstr(MachineInstr &MI);
  bool replaceImageHandle(MachineOperand &Op, MachineFunction &MF);
};
} // namespace

bool NVPTXReplaceImageHandles::run(MachineFunction &MF) {
  bool Changed = false;
  InstrsToRemove.clear();

  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      Changed |= processInstr(MI);

  // Now clean up any handle-access instructions
  // This is needed in debug mode when code cleanup passes are not executed,
  // but we need the handle access to be eliminated because they are not
  // valid instructions when image handles are disabled.
  for (MachineInstr *MI : InstrsToRemove) {
    unsigned DefReg = MI->getOperand(0).getReg();
    // Only these that are not used can be removed.
    if (MF.getRegInfo().use_nodbg_empty(DefReg))
      MI->eraseFromParent();
  }
  return Changed;
}

bool NVPTXReplaceImageHandles::processInstr(MachineInstr &MI) {
  MachineFunction &MF = *MI.getParent()->getParent();
  const MCInstrDesc &MCID = MI.getDesc();

  if (MCID.TSFlags & NVPTXII::IsTexFlag) {
    // This is a texture fetch, so operand 4 is a texref and operand 5 is
    // a samplerref
    MachineOperand &TexHandle = MI.getOperand(4);
    bool Changed = replaceImageHandle(TexHandle, MF);

    if (!(MCID.TSFlags & NVPTXII::IsTexModeUnifiedFlag)) {
      MachineOperand &SampHandle = MI.getOperand(5);
      Changed |= replaceImageHandle(SampHandle, MF);
    }

    return Changed;
  }
  if (MCID.TSFlags & NVPTXII::IsSuldMask) {
    unsigned VecSize =
        1 << (((MCID.TSFlags & NVPTXII::IsSuldMask) >> NVPTXII::IsSuldShift) -
              1);

    // For a surface load of vector size N, the Nth operand will be the surfref
    MachineOperand &SurfHandle = MI.getOperand(VecSize);

    return replaceImageHandle(SurfHandle, MF);
  }
  if (MCID.TSFlags & NVPTXII::IsSustFlag) {
    // This is a surface store, so operand 0 is a surfref
    MachineOperand &SurfHandle = MI.getOperand(0);

    return replaceImageHandle(SurfHandle, MF);
  }
  if (MCID.TSFlags & NVPTXII::IsSurfTexQueryFlag) {
    // This is a query, so operand 1 is a surfref/texref
    MachineOperand &Handle = MI.getOperand(1);

    return replaceImageHandle(Handle, MF);
  }

  return false;
}

bool NVPTXReplaceImageHandles::replaceImageHandle(MachineOperand &Op,
                                                  MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  NVPTXMachineFunctionInfo *MFI = MF.getInfo<NVPTXMachineFunctionInfo>();

  assert(Op.isReg() && "Handle is not in a reg?");

  // Which instruction defines the handle?
  MachineInstr &TexHandleDef = *MRI.getVRegDef(Op.getReg());

  switch (TexHandleDef.getOpcode()) {
  case NVPTX::LD_i64: {
    // The handle is a parameter value being loaded, replace with the
    // parameter symbol
    const auto &TM = static_cast<const NVPTXTargetMachine &>(MF.getTarget());
    if (TM.getDrvInterface() == NVPTX::CUDA)
      // For CUDA, we preserve the param loads coming from function arguments
      return false;

    assert(TexHandleDef.getOperand(7).isMCSymbol() && "Load is not a symbol!");
    MCSymbol *Sym = TexHandleDef.getOperand(7).getMCSymbol();
    InstrsToRemove.insert(&TexHandleDef);
    Op.ChangeToMCSymbol(Sym);
    MFI->addImageHandleSymbol(Sym);
    return true;
  }
  case NVPTX::texsurf_handles: {
    // The handle is a global variable, replace with the global variable name
    assert(TexHandleDef.getOperand(1).isGlobal() && "Load is not a global!");
    const GlobalValue *GV = TexHandleDef.getOperand(1).getGlobal();
    assert(GV->hasName() && "Global sampler must be named!");
    InstrsToRemove.insert(&TexHandleDef);
    Op.ChangeToGA(GV, 0);
    return true;
  }
  case NVPTX::nvvm_move_i64:
  case TargetOpcode::COPY: {
    bool Res = replaceImageHandle(TexHandleDef.getOperand(1), MF);
    if (Res)
      InstrsToRemove.insert(&TexHandleDef);
    return Res;
  }
  default:
    llvm_unreachable("Unknown instruction operating on handle");
  }
}

namespace {
class NVPTXReplaceImageHandlesLegacyPass : public MachineFunctionPass {
public:
  static char ID;
  NVPTXReplaceImageHandlesLegacyPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return NVPTXReplaceImageHandles().run(MF);
  }

  StringRef getPassName() const override {
    return "NVPTX Replace Image Handles";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

char NVPTXReplaceImageHandlesLegacyPass::ID = 0;

MachineFunctionPass *llvm::createNVPTXReplaceImageHandlesLegacyPass() {
  return new NVPTXReplaceImageHandlesLegacyPass();
}

PreservedAnalyses
NVPTXReplaceImageHandlesPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  if (!NVPTXReplaceImageHandles().run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
