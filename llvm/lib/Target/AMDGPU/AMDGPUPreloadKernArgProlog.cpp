//===- AMDGPUPreloadKernArgProlog.cpp - Preload KernArg Prolog ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass creates a backward compatibility layer for kernel argument
/// preloading in situations where code is compiled with kernel argument
/// preloading enabled but executed on hardware without firmware support for it.
///
/// To avoid recompilation, the pass inserts a block at the beginning of the
/// program that loads the kernel arguments into SGPRs using s_load
/// instructions. This sets up the registers exactly as they would be on systems
/// with compatible firmware.
///
/// This effectively creates two entry points for the kernel. Firmware that
/// supports the feature will automatically jump past the first 256 bytes of the
/// program, skipping the compatibility layer and directly starting execution on
/// the optimized code path.
///
/// This pass should be run as late as possible to prevent any optimizations
/// that might assume the padding is dead code or that the added prologue is a
/// true predecessor of the kernel entry block.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPreloadKernArgProlog.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-preload-kern-arg-prolog"

namespace {

// Used to build s_loads maping user SGPRs to kernel arguments
struct LoadConfig {
  unsigned Size;
  const TargetRegisterClass *RegClass;
  unsigned Opcode;
  Register LoadReg = Register();
};

class AMDGPUPreloadKernArgProlog {
public:
  AMDGPUPreloadKernArgProlog(MachineFunction &MF);

  bool run();

private:
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIMachineFunctionInfo &MFI;
  const SIInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  // Create a new block before the entry point to the kernel. Firmware that
  // supports preloading kernel arguments will automatically jump past this
  // block to the alternative kernel entry point.
  void createBackCompatBlock(unsigned NumKernArgPreloadSGPRs);

  // Add instructions to load kernel arguments into SGPRs.
  void addBackCompatLoads(MachineBasicBlock *BackCompatMBB,
                          Register KernArgSegmentPtr,
                          unsigned NumKernArgPreloadSGPRs);
};

class AMDGPUPreloadKernArgPrologLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreloadKernArgPrologLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU Preload Kernel Arguments Prolog";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char AMDGPUPreloadKernArgPrologLegacy::ID = 0;

INITIALIZE_PASS(AMDGPUPreloadKernArgPrologLegacy, DEBUG_TYPE,
                "AMDGPU Preload Kernel Arguments Prolog", false, false)

char &llvm::AMDGPUPreloadKernArgPrologLegacyID =
    AMDGPUPreloadKernArgPrologLegacy::ID;

FunctionPass *llvm::createAMDGPUPreloadKernArgPrologLegacyPass() {
  return new AMDGPUPreloadKernArgPrologLegacy();
}

bool AMDGPUPreloadKernArgPrologLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return AMDGPUPreloadKernArgProlog(MF).run();
}

AMDGPUPreloadKernArgProlog::AMDGPUPreloadKernArgProlog(MachineFunction &MF)
    : MF(MF), ST(MF.getSubtarget<GCNSubtarget>()),
      MFI(*MF.getInfo<SIMachineFunctionInfo>()), TII(*ST.getInstrInfo()),
      TRI(*ST.getRegisterInfo()) {}

bool AMDGPUPreloadKernArgProlog::run() {
  if (!ST.hasKernargPreload())
    return false;

  unsigned NumKernArgPreloadSGPRs = MFI.getNumKernargPreloadedSGPRs();
  if (!NumKernArgPreloadSGPRs)
    return false;

  createBackCompatBlock(NumKernArgPreloadSGPRs);
  return true;
}

void AMDGPUPreloadKernArgProlog::createBackCompatBlock(
    unsigned NumKernArgPreloadSGPRs) {
  auto KernelEntryMBB = MF.begin();
  MachineBasicBlock *BackCompatMBB = MF.CreateMachineBasicBlock();
  MF.insert(KernelEntryMBB, BackCompatMBB);

  assert(MFI.getUserSGPRInfo().hasKernargSegmentPtr() &&
         "Kernel argument segment pointer register not set.");
  Register KernArgSegmentPtr = MFI.getArgInfo().KernargSegmentPtr.getRegister();
  BackCompatMBB->addLiveIn(KernArgSegmentPtr);

  // Load kernel arguments to SGPRs
  addBackCompatLoads(BackCompatMBB, KernArgSegmentPtr, NumKernArgPreloadSGPRs);

  // Wait for loads to complete
  AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST.getCPU());
  unsigned Waitcnt =
      AMDGPU::encodeWaitcnt(IV, getVmcntBitMask(IV), getExpcntBitMask(IV), 0);
  BuildMI(BackCompatMBB, DebugLoc(), TII.get(AMDGPU::S_WAITCNT))
      .addImm(Waitcnt);

  // Branch to kernel start
  BuildMI(BackCompatMBB, DebugLoc(), TII.get(AMDGPU::S_BRANCH))
      .addMBB(&*KernelEntryMBB);
  BackCompatMBB->addSuccessor(&*KernelEntryMBB);

  // Create a new basic block for padding to 256 bytes
  MachineBasicBlock *PadMBB = MF.CreateMachineBasicBlock();
  MF.insert(++BackCompatMBB->getIterator(), PadMBB);
  PadMBB->setAlignment(Align(256));
  PadMBB->addSuccessor(&*KernelEntryMBB);
}

/// Find the largest possible load size that fits with SGPR alignment
static LoadConfig getLoadParameters(const TargetRegisterInfo &TRI,
                                    Register KernArgPreloadSGPR,
                                    unsigned NumKernArgPreloadSGPRs) {
  static constexpr LoadConfig Configs[] = {
      {8, &AMDGPU::SReg_256RegClass, AMDGPU::S_LOAD_DWORDX8_IMM},
      {4, &AMDGPU::SReg_128RegClass, AMDGPU::S_LOAD_DWORDX4_IMM},
      {2, &AMDGPU::SReg_64RegClass, AMDGPU::S_LOAD_DWORDX2_IMM}};

  for (const auto &Config : Configs) {
    if (NumKernArgPreloadSGPRs >= Config.Size) {
      Register LoadReg = TRI.getMatchingSuperReg(KernArgPreloadSGPR,
                                                 AMDGPU::sub0, Config.RegClass);
      if (LoadReg) {
        LoadConfig C(Config);
        C.LoadReg = LoadReg;
        return C;
      }
    }
  }

  // Fallback to a single register
  return LoadConfig{1, &AMDGPU::SReg_32RegClass, AMDGPU::S_LOAD_DWORD_IMM,
                    KernArgPreloadSGPR};
}

void AMDGPUPreloadKernArgProlog::addBackCompatLoads(
    MachineBasicBlock *BackCompatMBB, Register KernArgSegmentPtr,
    unsigned NumKernArgPreloadSGPRs) {
  Register KernArgPreloadSGPR = MFI.getArgInfo().FirstKernArgPreloadReg;
  unsigned Offset = 0;
  // Fill all user SGPRs used for kernarg preloading with sequential data from
  // the kernarg segment
  while (NumKernArgPreloadSGPRs > 0) {
    LoadConfig Config =
        getLoadParameters(TRI, KernArgPreloadSGPR, NumKernArgPreloadSGPRs);

    BuildMI(BackCompatMBB, DebugLoc(), TII.get(Config.Opcode), Config.LoadReg)
        .addReg(KernArgSegmentPtr)
        .addImm(Offset)
        .addImm(0);

    Offset += 4 * Config.Size;
    KernArgPreloadSGPR = KernArgPreloadSGPR.asMCReg() + Config.Size;
    NumKernArgPreloadSGPRs -= Config.Size;
  }
}

PreservedAnalyses
AMDGPUPreloadKernArgPrologPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &) {
  if (!AMDGPUPreloadKernArgProlog(MF).run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
