//===- NVPTXAddressFolder.cpp - Fold symbol addresses into memory ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SelectionDAG folds a symbol address (a kernel parameter, global variable, or
// external symbol) directly into the address operand of a memory access, but
// only within a single basic block. When the address is carried across a block
// boundary in a register it is materialized with a generic `mov`
// (MOV_B{32,64}_sym) and the accesses become register-relative:
//
//     mov.b64       %rd1, kernel_param_0;
//     ld.param.b64  %rd2, [%rd1];
//     ld.param.b64  %rd3, [%rd1+8];
//
// This pass folds the symbol back into those address operands, eliminating the
// redundant address arithmetic (and the `mov` itself once no use remains):
//
//     ld.param.b64  %rd2, [kernel_param_0];
//     ld.param.b64  %rd3, [kernel_param_0+8];
//
// Shared-memory accesses are left alone: `mov`s of shared symbols are
// deliberately kept CSE-able rather than duplicated into their uses, as
// rematerializing them has caused performance regressions before (see
// MovSymInst in NVPTXInstrInfo.td).
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

// Try to fold the definition of \p Addr into \p MI's address operand. The
// defining instruction is erased once it has no uses left; it is kept for any
// remaining use, e.g. because the address also feeds arithmetic or escapes.
static bool foldAddress(MachineInstr &MI, MachineOperand &Addr,
                        MachineRegisterInfo &MRI) {
  assert(Addr.isReg() && "Expected an address register");
  assert(Addr.getReg().isVirtual() && "Expected a virtual address register");

  MachineInstr *Mov = MRI.getVRegDef(Addr.getReg());
  if (!Mov || (Mov->getOpcode() != NVPTX::MOV_B32_sym &&
               Mov->getOpcode() != NVPTX::MOV_B64_sym))
    return false;

  const MachineOperand &Sym = Mov->getOperand(1);
  if (!Sym.isGlobal() && !Sym.isSymbol())
    return false;

  // The accessed address space must be known and must not be shared.
  const int AddrSpaceIdx =
      NVPTX::getNamedOperandIdx(MI.getOpcode(), NVPTX::OpName::addsp);
  if (AddrSpaceIdx < 0)
    return false;
  const auto AddrSpace = MI.getOperand(AddrSpaceIdx).getImm();
  if (AddrSpace == NVPTX::AddressSpace::Shared ||
      AddrSpace == NVPTX::AddressSpace::SharedCluster)
    return false;

  if (Sym.isGlobal()) {
    Addr.ChangeToGA(Sym.getGlobal(), Sym.getOffset(), Sym.getTargetFlags());
  } else {
    Addr.ChangeToES(Sym.getSymbolName(), Sym.getTargetFlags());
  }

  if (MRI.use_empty(Mov->getOperand(0).getReg()))
    Mov->eraseFromParent();

  return true;
}

static bool foldAddresses(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : make_early_inc_range(MBB))
      if (MI.mayLoadOrStore()) {
        const int AddrIdx =
            NVPTX::getNamedOperandIdx(MI.getOpcode(), NVPTX::OpName::addr);
        if (AddrIdx >= 0 && MI.getOperand(AddrIdx).isReg())
          Changed |= foldAddress(MI, MI.getOperand(AddrIdx), MRI);
      }

  return Changed;
}

/// ----------------------------------------------------------------------------
///                       Pass (Manager) Boilerplate
/// ----------------------------------------------------------------------------

namespace {
struct NVPTXAddressFolderPass : public MachineFunctionPass {
  static char ID;
  NVPTXAddressFolderPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

char NVPTXAddressFolderPass::ID = 0;

INITIALIZE_PASS(NVPTXAddressFolderPass, "nvptx-address-folder",
                "NVPTX Address Folder", false, false)

bool NVPTXAddressFolderPass::runOnMachineFunction(MachineFunction &MF) {
  return foldAddresses(MF);
}

MachineFunctionPass *llvm::createNVPTXAddressFolderPass() {
  return new NVPTXAddressFolderPass();
}
