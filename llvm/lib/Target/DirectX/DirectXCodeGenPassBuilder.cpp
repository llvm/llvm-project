//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
#include "DXILSignatureAnalysis.h"
#include "DXILWriter/DXILWriterPass.h"
#include "DirectX.h"
#include "DirectXAsmPrinter.h"
#include "DirectXTargetMachine.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Utils/StripConvergenceIntrinsics.h"

using namespace llvm;

namespace {

class DirectXCodeGenPassBuilder : public CodeGenPassBuilder {
  DirectXTargetMachine &getTM() const {
    return static_cast<DirectXTargetMachine &>(TM);
  }

public:
  explicit DirectXCodeGenPassBuilder(DirectXTargetMachine &TM,
                                     const CGPassBuilderOption &Opts,
                                     PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

protected:
  Error buildPipelineImpl(PassManagerWrapper &PMW, ModuleAnalysisManager &MAM,
                          raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
                          CodeGenFileType FileType, MCContext &Ctx) override;
  void addCodeGenPrepare(PassManagerWrapper &PMW) override;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;
};

} // namespace

void DirectXCodeGenPassBuilder::addCodeGenPrepare(PassManagerWrapper &PMW) {
  CodeGenOptLevel OptLevel = getOptLevel();

  addFunctionPass(StripConvergenceIntrinsicsPass(), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(DXILFinalizeLinkagePass(), PMW);
  if (OptLevel != CodeGenOptLevel::None)
    addModulePass(GlobalDCEPass(), PMW);
  addModulePass(DXILMemIntrinsicsPass(), PMW);
  addModulePass(DXILCBufferAccessPass(), PMW);
  addFunctionPass(DXILRemoveUnusedResourcesPass(), PMW);
  addFunctionPass(DXILResourceAccessPass(), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(DXILIntrinsicExpansionPass(), PMW);
  addModulePass(DXILDataScalarizationPass(), PMW);
  if (getTM().getTargetTriple().getOSVersion() < VersionTuple(6, 9)) {
    ScalarizerPassOptions DxilScalarOptions;
    DxilScalarOptions.ScalarizeLoadStore = true;
    addFunctionPass(ScalarizerPass(DxilScalarOptions), PMW);
    flushFPMsToMPM(PMW);
  }
  addModulePass(DXILFlattenArraysPass(), PMW);
  addFunctionPass(DXILForwardHandleAccessesPass(), PMW);
  if (OptLevel != CodeGenOptLevel::None)
    addFunctionPass(DSEPass(), PMW);
  addFunctionPass(DXILLegalizePass(), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(DXILResourceImplicitBindingPass(), PMW);
  addModulePass(DXILTranslateMetadataPass(), PMW);
  addModulePass(DXILPostOptimizationValidationPass(), PMW);
  addModulePass(DXILOpLoweringPass(), PMW);
  addModulePass(DXILPrepareModulePass(), PMW);
  addModulePass(DXILDebugInfoPass(), PMW);
}

void DirectXCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  addModulePass(DXILAsmPrinterBeginPass(), PMW, /*Force=*/true);
}

void DirectXCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  addMachineFunctionPass(DXILAsmPrinterPass(), PMW);
}

void DirectXCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  addModulePass(DXILAsmPrinterEndPass(), PMW, /*Force=*/true);
}

Error DirectXCodeGenPassBuilder::buildPipelineImpl(PassManagerWrapper &PMW,
                                                   ModuleAnalysisManager &MAM,
                                                   raw_pwrite_stream &Out,
                                                   raw_pwrite_stream *DwoOut,
                                                   CodeGenFileType FileType,
                                                   MCContext &Ctx) {
  addModulePass(RequireAnalysisPass<MachineModuleAnalysis, Module>(), PMW,
                /*Force=*/true);
  addCodeGenPrepare(PMW);
  flushFPMsToMPM(PMW);

  switch (FileType) {
  case CodeGenFileType::AssemblyFile:
    addModulePass(DXILPrettyPrinterPass(Out), PMW, /*Force=*/true);
    break;
  case CodeGenFileType::ObjectFile:
    if (CompletesPipeline) {
      addModulePass(DXILEmbedderPass(), PMW, /*Force=*/true);
      // We embed the other DXContainer globals after embedding DXIL so that the
      // globals don't pollute the DXIL.
      addModulePass(DXILContainerGlobalsPass(), PMW, /*Force=*/true);
      addModulePass(DXILContainerPDBPass(), PMW, /*Force=*/true);

      if (auto Err = registerAsmPrinter(MAM, Out, DwoOut, FileType, Ctx))
        return Err;

      addAsmPrinterBegin(PMW);
      addAsmPrinter(PMW);
      flushFPMsToMPM(PMW, /*FreeMachineFunctions=*/true);
      addAsmPrinterEnd(PMW);
    } else {
      addModulePass(DXILWriterPass(Out), PMW, /*Force=*/true);
    }
    break;
  case CodeGenFileType::Null:
    break;
  }

  return Error::success();
}

void DirectXTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "DirectXPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error DirectXTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = DirectXCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
