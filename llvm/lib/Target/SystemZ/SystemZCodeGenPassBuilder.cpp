//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/EarlyIfConversion.h"
#include "llvm/CodeGen/MachineCombiner.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"

using namespace llvm;

extern cl::opt<bool> EnableMachineCombinerPass;

namespace {

class SystemZCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  SystemZTargetMachine &getTM() const {
    return static_cast<SystemZTargetMachine &>(TM);
  }

public:
  explicit SystemZCodeGenPassBuilder(SystemZTargetMachine &TM,
                                     const CGPassBuilderOption &Opts,
                                     PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addILPOpts(PassManagerWrapper &PMW) override;
  void addPreRegAlloc(PassManagerWrapper &PMW) override;
  void addPostRewrite(PassManagerWrapper &PMW) override;
  void addPostRegAlloc(PassManagerWrapper &PMW) override;
  void addPreSched2(PassManagerWrapper &PMW) override;
  void addPreEmitPass(PassManagerWrapper &PMW) override;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;
};

void SystemZCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add SystemZAlignGlobalsPass(
    // TODO(boomanaiden154): Add SystemZTDCPass
    addFunctionPass(LoopDataPrefetchPass(), PMW);
  }

  addFunctionPass(AtomicExpandPass(getTM()), PMW);

  Base::addIRPasses(PMW);
}

Error SystemZCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addMachineFunctionPass(SystemZISelDAGToDAGPass(getTM(), getOptLevel()), PMW);

  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add SystemZLDCleanupPass
  }

  return Error::success();
}

void SystemZCodeGenPassBuilder::addILPOpts(PassManagerWrapper &PMW) {
  addMachineFunctionPass(EarlyIfConverterPass(), PMW);

  if (EnableMachineCombinerPass)
    addMachineFunctionPass(MachineCombinerPass(), PMW);
}

void SystemZCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SystemZCopyPhysRegsPass
}

void SystemZCodeGenPassBuilder::addPostRewrite(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SystemZPostRewritePass
}

void SystemZCodeGenPassBuilder::addPostRegAlloc(PassManagerWrapper &PMW) {
  // PostRewrite needs to be run at -O0 also (in which case addPostRewrite()
  // is not called).
  if (getOptLevel() == CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add SystemZPostRewritePass
  }
}

void SystemZCodeGenPassBuilder::addPreSched2(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add IfConverter
  }
}

void SystemZCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  // Do instruction shortening before compare elimination because some
  // vector instructions will be shortened into opcodes that compare
  // elimination recognizes.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add SystemZShortenInstPass
  }

  // We eliminate comparisons here rather than earlier because some
  // transformations can change the set of available CC values and we
  // generally want those transformations to have priority.  This is
  // especially true in the commonest case where the result of the comparison
  // is used by a single in-range branch instruction, since we will then
  // be able to fuse the compare and the branch instead.
  //
  // For example, two-address NILF can sometimes be converted into
  // three-address RISBLG.  NILF produces a CC value that indicates whether
  // the low word is zero, but RISBLG does not modify CC at all.  On the
  // other hand, 64-bit ANDs like NILL can sometimes be converted to RISBG.
  // The CC value produced by NILL isn't useful for our purposes, but the
  // value produced by RISBG can be used for any comparison with zero
  // (not just equality).  So there are some transformations that lose
  // CC values (while still being worthwhile) and others that happen to make
  // the CC result more useful than it was originally.
  //
  // Another reason is that we only want to use BRANCH ON COUNT in cases
  // where we know that the count register is not going to be spilled.
  //
  // Doing it so late makes it more likely that a register will be reused
  // between the comparison and the branch, but it isn't clear whether
  // preventing that would be a win or not.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add SystemZElimComparePass
  }
  // TODO(boomanaiden154): Add SystemZLongBranchPass

  // Do final scheduling after all other optimizations, to get an
  // optimal input for the decoder (branch relaxation must happen
  // after block placement).
  if (getOptLevel() != CodeGenOptLevel::None)
    addMachineFunctionPass(PostMachineSchedulerPass(&getTM()), PMW);
}

void SystemZCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SystemZAsmPrinterBegin
}

void SystemZCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SystemZAsmPrinter
}

void SystemZCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SystemZAsmPrinterEnd
}

} // namespace

void SystemZTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "SystemZPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error SystemZTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = SystemZCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
