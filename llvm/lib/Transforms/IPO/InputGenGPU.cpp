//===-- InputGenGPU.cpp - InputGen GPU instrumentation pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InputGenGPU.h"
#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "inputgen-gpu"

static cl::list<std::string> InputGenGPURuntimeBitcodes(
    "inputgen-gpu-runtime-bitcode",
    cl::desc("InputGen GPU runtime bitcode file; may be repeated"),
    cl::ZeroOrMore);

namespace {

class InputGenGPUConfig final : public InstrumentationConfig {
  void populate(InstrumentorIRBuilderTy &IIRB) override {
    InstrumentationConfig::populate(IIRB);

    RuntimePrefix->setString("__ig_");
    HostEnabled->setBool(false);
    GPUEnabled->setBool(true);
    SmallVector<StringRef> RuntimeBitcodeRefs;
    for (StringRef RuntimeBitcode : InputGenGPURuntimeBitcodes)
      RuntimeBitcodeRefs.push_back(RuntimeBitcode);
    RuntimeBitcodes->setStringList(RuntimeBitcodeRefs);
    InlineRuntimeEagerly->setBool(false);

    for (auto &ChoiceMap : IChoices) {
      for (auto &ChoiceIt : ChoiceMap) {
        auto *IO = ChoiceIt.second;
        IO->Enabled = false;
        IO->Filter = "";
        for (IRTArg &Arg : IO->IRTArgs)
          Arg.Enabled = false;
      }
    }

    auto *PostLoad =
        IChoices[InstrumentationLocation::INSTRUCTION_POST].lookup("load");
    if (!PostLoad)
      return;

    PostLoad->Enabled = true;
    for (IRTArg &Arg : PostLoad->IRTArgs) {
      Arg.Enabled = Arg.Name == "value" || Arg.Name == "value_size" ||
                    Arg.Name == "value_type_id" || Arg.Name == "id";
    }
  }
};

} // end anonymous namespace

PreservedAnalyses InputGenGPUPass::run(Module &M, ModuleAnalysisManager &MAM) {
  InputGenGPUConfig IConf;
  InstrumentorIRBuilderTy IIRB(M);
  return InstrumentorPass(/*FS=*/nullptr, &IConf, &IIRB).run(M, MAM);
}
