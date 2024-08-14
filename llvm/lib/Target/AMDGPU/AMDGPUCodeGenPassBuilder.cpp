//===- lib/Target/AMDGPU/AMDGPUCodeGenPassBuilder.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCodeGenPassBuilder.h"
#include "AMDGPU.h"
#include "AMDGPUISelDAGToDAG.h"
#include "AMDGPUPerfHintAnalysis.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnifyDivergentExitNodes.h"
#include "SIFixSGPRCopies.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Transforms/Scalar/FlattenCFG.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"

using namespace llvm;

AMDGPUCodeGenPassBuilder::AMDGPUCodeGenPassBuilder(
    GCNTargetMachine &TM, const CGPassBuilderOption &Opts,
    PassInstrumentationCallbacks *PIC)
    : CodeGenPassBuilder(TM, Opts, PIC) {
  Opt.RequiresCodeGenSCCOrder = true;
  // Exceptions and StackMaps are not supported, so these passes will never do
  // anything.
  // Garbage collection is not supported.
  disablePass<StackMapLivenessPass, FuncletLayoutPass,
              ShadowStackGCLoweringPass>();
}

void AMDGPUCodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  const bool LateCFGStructurize = AMDGPUTargetMachine::EnableLateStructurizeCFG;
  const bool DisableStructurizer = AMDGPUTargetMachine::DisableStructurizer;
  const bool EnableStructurizerWorkarounds =
      AMDGPUTargetMachine::EnableStructurizerWorkarounds;

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(FlattenCFGPass());

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(SinkingPass());

  addPass(AMDGPULateCodeGenPreparePass(TM));

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.

  addPass(AMDGPUUnifyDivergentExitNodesPass());

  if (!LateCFGStructurize && !DisableStructurizer) {
    if (EnableStructurizerWorkarounds) {
      addPass(FixIrreduciblePass());
      addPass(UnifyLoopExitsPass());
    }

    addPass(StructurizeCFGPass(/*SkipUniformRegions=*/false));
  }

  addPass(AMDGPUAnnotateUniformValuesPass());

  if (!LateCFGStructurize && !DisableStructurizer) {
    addPass(SIAnnotateControlFlowPass(TM));

    // TODO: Move this right after structurizeCFG to avoid extra divergence
    // analysis. This depends on stopping SIAnnotateControlFlow from making
    // control flow modifications.
    addPass(AMDGPURewriteUndefForPHIPass());
  }

  addPass(LCSSAPass());

  if (TM.getOptLevel() > CodeGenOptLevel::Less)
    addPass(AMDGPUPerfHintAnalysisPass(TM));

  // FIXME: Why isn't this queried as required from AMDGPUISelDAGToDAG, and why
  // isn't this in addInstSelector?
  addPass(RequireAnalysisPass<UniformityInfoAnalysis, Function>());
}

void AMDGPUCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                             CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error AMDGPUCodeGenPassBuilder::addInstSelector(AddMachinePass &addPass) const {
  addPass(AMDGPUISelDAGToDAGPass(TM));
  addPass(SIFixSGPRCopiesPass());
  addPass(SILowerI1CopiesPass());
  return Error::success();
}
