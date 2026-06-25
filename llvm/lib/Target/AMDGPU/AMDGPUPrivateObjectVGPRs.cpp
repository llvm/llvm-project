//===-- AMDGPUPrivateObjectVGPRs.cpp - Lower VGPR-as-memory accesses ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lowers the constant-index SI_VGPR_FRAME_{LOAD,STORE} pseudos for "VGPR as
/// memory" objects (addrspace(13)) into register copies to/from the block of
/// physical VGPRs backing the file: a load is a COPY from the file register, a
/// store a COPY to it.
///
/// The file is a fixed block of VGPRs (SIRegisterInfo::getVGPRMemoryFile)
/// reserved out of allocation (getReservedRegs) and counted in the VGPR usage
/// (AMDGPUResourceUsageAnalysis). It sits just above the ABI inputs at a base
/// AMDGPULowerModuleVGPRs shares across the call graph (so an address resolves
/// to the same registers everywhere), low enough to cost only its own size
/// rather than pinning occupancy. This pass runs after register allocation;
/// until then the pseudos behave as opaque memory operations, so allocation is
/// free to use any other register for the surrounding code.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPrivateObjectVGPRs.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-private-object-vgprs"

namespace {

// These two switches must list the same widths as the SI_VGPR_FRAME_{LOAD,
// STORE}_B* `foreach` in SIInstructions.td.
static bool isVGPRFrameLoad(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::SI_VGPR_FRAME_LOAD_B32:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B64:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B96:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B128:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B160:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B192:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B224:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B256:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B288:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B320:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B352:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B384:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B512:
  case AMDGPU::SI_VGPR_FRAME_LOAD_B1024:
    return true;
  default:
    return false;
  }
}

static bool isVGPRFrameStore(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::SI_VGPR_FRAME_STORE_B32:
  case AMDGPU::SI_VGPR_FRAME_STORE_B64:
  case AMDGPU::SI_VGPR_FRAME_STORE_B96:
  case AMDGPU::SI_VGPR_FRAME_STORE_B128:
  case AMDGPU::SI_VGPR_FRAME_STORE_B160:
  case AMDGPU::SI_VGPR_FRAME_STORE_B192:
  case AMDGPU::SI_VGPR_FRAME_STORE_B224:
  case AMDGPU::SI_VGPR_FRAME_STORE_B256:
  case AMDGPU::SI_VGPR_FRAME_STORE_B288:
  case AMDGPU::SI_VGPR_FRAME_STORE_B320:
  case AMDGPU::SI_VGPR_FRAME_STORE_B352:
  case AMDGPU::SI_VGPR_FRAME_STORE_B384:
  case AMDGPU::SI_VGPR_FRAME_STORE_B512:
  case AMDGPU::SI_VGPR_FRAME_STORE_B1024:
    return true;
  default:
    return false;
  }
}

class AMDGPUPrivateObjectVGPRs {
public:
  bool run(MachineFunction &MF);
};

class AMDGPUPrivateObjectVGPRsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrivateObjectVGPRsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return AMDGPUPrivateObjectVGPRs().run(MF);
  }

  StringRef getPassName() const override {
    return "AMDGPU Private Object VGPRs";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUPrivateObjectVGPRsLegacy, DEBUG_TYPE,
                "AMDGPU Private Object VGPRs", false, false)

char AMDGPUPrivateObjectVGPRsLegacy::ID = 0;

char &llvm::AMDGPUPrivateObjectVGPRsID = AMDGPUPrivateObjectVGPRsLegacy::ID;

PreservedAnalyses
AMDGPUPrivateObjectVGPRsPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  if (!AMDGPUPrivateObjectVGPRs().run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}

bool AMDGPUPrivateObjectVGPRs::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // The file is a fixed block of reserved physical VGPRs (getVGPRMemoryFile):
  // exempt from liveness, needing no explicit def, and at the same (shared)
  // registers across the call graph. (Plain locals rather than a structured
  // binding, which cannot be captured by the lambdas below in C++17.)
  std::pair<unsigned, unsigned> File = TRI->getVGPRMemoryFile(MF);
  const unsigned BaseIdx = File.first;
  const unsigned FileDwords = File.second;
  if (FileDwords == 0)
    return false;

  const TargetRegisterClass &VGPR32 = AMDGPU::VGPR_32RegClass;

  // The file lives in low, caller-saved VGPRs. AMDGPULowerModuleVGPRs diagnoses
  // calls that escape the group at the IR level, but later passes (e.g.
  // AtomicExpand, CodeGenPrepare) can introduce libcalls, and inline asm naming
  // a file register is not seen there at all. Both would clobber the file, so
  // catch them here, now that the reserved registers and machine calls are
  // final.
  LLVMContext &Ctx = MF.getFunction().getContext();
  auto FileOverlaps = [&](Register Reg) {
    for (unsigned I = 0; I != FileDwords; ++I)
      if (TRI->regsOverlap(Reg, VGPR32.getRegister(BaseIdx + I)))
        return true;
    return false;
  };
  auto RegMaskClobbersFile = [&](const MachineOperand &MO) {
    for (unsigned I = 0; I != FileDwords; ++I)
      if (MO.clobbersPhysReg(VGPR32.getRegister(BaseIdx + I)))
        return true;
    return false;
  };
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isInlineAsm()) {
        // A clobber surfaces either as an explicit physical-register def or,
        // for some forms, as a register-mask operand; check both.
        for (const MachineOperand &MO : MI.operands())
          if ((MO.isReg() && MO.getReg().isPhysical() && MO.isDef() &&
               FileOverlaps(MO.getReg())) ||
              (MO.isRegMask() && RegMaskClobbersFile(MO))) {
            Ctx.diagnose(DiagnosticInfoUnsupported(
                MF.getFunction(),
                "inline asm clobbers a 'VGPR as memory' reserved register",
                MI.getDebugLoc()));
            break;
          }
        continue;
      }
      // A call clobbers caller-saved VGPRs, including the file, unless the
      // callee reserves the same file: an in-group member (which carries the
      // size attribute) or this function itself (self-recursion). Anything else
      // - an out-of-group/external callee, or an indirect call with no
      // resolvable callee - does not preserve it. AMDGPULowerModuleVGPRs
      // catches IR-level escapes; this also covers calls introduced after it
      // (e.g. expanded libcalls) and indirect machine calls it could not see.
      if (MI.isCall()) {
        const MachineOperand *CalleeOp =
            TII->getNamedOperand(MI, AMDGPU::OpName::callee);
        const auto *Callee =
            CalleeOp && CalleeOp->isGlobal()
                ? dyn_cast<Function>(
                      CalleeOp->getGlobal()->stripPointerCastsAndAliases())
                : nullptr;
        if (Callee == &MF.getFunction() ||
            (Callee && Callee->hasFnAttribute("amdgpu-vgpr-memory-size")))
          continue;
        Ctx.diagnose(DiagnosticInfoUnsupported(
            MF.getFunction(),
            "call to a function that clobbers the 'VGPR as memory' reserved "
            "file",
            MI.getDebugLoc()));
      }
    }
  }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      unsigned Opc = MI.getOpcode();
      bool IsLoad = isVGPRFrameLoad(Opc);
      if (!IsLoad && !isVGPRFrameStore(Opc))
        continue;

      const DebugLoc &DL = MI.getDebugLoc();
      unsigned Dword = MI.getOperand(1).getImm();
      Register Data = MI.getOperand(0).getReg();
      unsigned AccessDwords = TRI->getRegSizeInBits(Data, MRI) / 32;

      // Bounds-checked at pseudo creation (LowerLoadStoreVGPR); never name a
      // register outside the reserved file.
      assert(Dword + AccessDwords <= FileDwords &&
             "VGPR-as-memory access outside the reserved file");

      // Copy the access dword-by-dword between the data (sub)registers and the
      // file registers. Doing it per dword rather than as one tuple COPY avoids
      // needing an aligned physical VGPR tuple for the file slice, which can
      // start on an odd register on targets that require aligned tuples.
      for (unsigned I = 0; I != AccessDwords; ++I) {
        MCRegister FileReg = VGPR32.getRegister(BaseIdx + Dword + I);
        Register DataReg =
            AccessDwords == 1
                ? Data
                : Register(TRI->getSubReg(
                      Data, SIRegisterInfo::getSubRegFromChannel(I)));
        if (IsLoad)
          BuildMI(MBB, MI, DL, TII->get(TargetOpcode::COPY), DataReg)
              .addReg(FileReg);
        else
          BuildMI(MBB, MI, DL, TII->get(TargetOpcode::COPY), FileReg)
              .addReg(DataReg);
      }

      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}
