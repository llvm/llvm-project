//===- EmbedBitcodePass.cpp - Pass that embeds the bitcode into a global---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/EmbedBitcodePass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/ThinLTOBitcodeWriter.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <string>

using namespace llvm;

static bool isFrontendEmbeddedBitcodeGlobal(const GlobalVariable *GV) {
  if (!GV || !GV->hasSection())
    return false;
  StringRef Sec = GV->getSection();
  return Sec == ".llvmbc" || Sec == "__LLVM,__bitcode";
}

PreservedAnalyses EmbedBitcodePass::run(Module &M, ModuleAnalysisManager &AM) {
  if (M.getGlobalVariable("llvm.embedded.object", /*AllowInternal=*/true))
    reportFatalUsageError("Can only embed the module once");

  GlobalVariable *PrelinkEmbeddedModule =
      M.getGlobalVariable("llvm.embedded.module", /*AllowInternal=*/true);
  if (PrelinkEmbeddedModule &&
      !isFrontendEmbeddedBitcodeGlobal(PrelinkEmbeddedModule))
    reportFatalUsageError("Can only embed the module once");

  Triple T(M.getTargetTriple());
  if (T.getObjectFormat() != Triple::ELF && T.getObjectFormat() != Triple::COFF)
    reportFatalUsageError("EmbedBitcode pass currently only supports COFF and "
                          "ELF object formats");

  GlobalVariable *PrelinkCmdline =
      PrelinkEmbeddedModule
          ? M.getGlobalVariable("llvm.cmdline", /*AllowInternal=*/true)
          : nullptr;
  SmallVector<GlobalValue *, 2> DetachedFrontendGlobals;
  if (PrelinkEmbeddedModule || PrelinkCmdline) {
    SmallVector<GlobalValue *, 4> UsedGlobals;
    if (GlobalVariable *Used =
            collectUsedGlobalVariables(M, UsedGlobals, /*CompilerUsed=*/true)) {
      SmallVector<Constant *, 4> KeptUsed;
      Type *ElemTy = Used->getValueType()->getArrayElementType();
      for (GlobalValue *GV : UsedGlobals) {
        if (GV != PrelinkEmbeddedModule && GV != PrelinkCmdline)
          KeptUsed.push_back(
              ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, ElemTy));
      }
      Used->eraseFromParent();
      if (!KeptUsed.empty()) {
        ArrayType *ATy = ArrayType::get(ElemTy, KeptUsed.size());
        auto *NewUsed = new GlobalVariable(
            M, ATy, false, GlobalValue::AppendingLinkage,
            ConstantArray::get(ATy, KeptUsed), "llvm.compiler.used");
        NewUsed->setSection("llvm.metadata");
      }
    }
    if (PrelinkEmbeddedModule) {
      PrelinkEmbeddedModule->removeFromParent();
      DetachedFrontendGlobals.push_back(PrelinkEmbeddedModule);
    }
    if (PrelinkCmdline) {
      PrelinkCmdline->removeFromParent();
      DetachedFrontendGlobals.push_back(PrelinkCmdline);
    }
  }

  std::string Data;
  raw_string_ostream OS(Data);
  if (IsThinLTO)
    ThinLTOBitcodeWriterPass(OS, /*ThinLinkOS=*/nullptr).run(M, AM);
  else
    BitcodeWriterPass(OS, /*ShouldPreserveUseListOrder=*/false, EmitLTOSummary)
        .run(M, AM);

  embedBufferInModule(M, MemoryBufferRef(Data, "ModuleData"), ".llvm.lto");

  if (!DetachedFrontendGlobals.empty()) {
    for (GlobalValue *GV : DetachedFrontendGlobals)
      M.insertGlobalVariable(M.global_end(), cast<GlobalVariable>(GV));
    appendToCompilerUsed(M, DetachedFrontendGlobals);
  }

  return PreservedAnalyses::none();
}
