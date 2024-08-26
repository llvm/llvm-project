//===- DXILPrettyPrinter.cpp - Print resources for textual DXIL -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILPrettyPrinter.h"
#include "DXILResourceAnalysis.h"
#include "DirectX.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void prettyPrintResources(raw_ostream &OS,
                                 const dxil::Resources &MDResources) {
  // Column widths are arbitrary but match the widths DXC uses.
  OS << ";\n; Resource Bindings:\n;\n";
  OS << formatv("; {0,-30} {1,10} {2,7} {3,11} {4,7} {5,14} {6,16}\n", "Name",
                "Type", "Format", "Dim", "ID", "HLSL Bind", "Count");
  OS << formatv(
      "; {0,-+30} {1,-+10} {2,-+7} {3,-+11} {4,-+7} {5,-+14} {6,-+16}\n", "",
      "", "", "", "", "", "");

  if (MDResources.hasCBuffers())
    MDResources.printCBuffers(OS);
  if (MDResources.hasUAVs())
    MDResources.printUAVs(OS);

  OS << ";\n";
}

PreservedAnalyses DXILPrettyPrinterPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  const dxil::Resources &MDResources = MAM.getResult<DXILResourceMDAnalysis>(M);
  prettyPrintResources(OS, MDResources);
  return PreservedAnalyses::all();
}

namespace {
class DXILPrettyPrinterLegacy : public llvm::ModulePass {
  raw_ostream &OS; // raw_ostream to print to.

public:
  static char ID;
  DXILPrettyPrinterLegacy() : ModulePass(ID), OS(dbgs()) {
    initializeDXILPrettyPrinterLegacyPass(*PassRegistry::getPassRegistry());
  }

  explicit DXILPrettyPrinterLegacy(raw_ostream &O) : ModulePass(ID), OS(O) {
    initializeDXILPrettyPrinterLegacyPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "DXIL Metadata Pretty Printer";
  }

  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<DXILResourceMDWrapper>();
  }
};
} // namespace

char DXILPrettyPrinterLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(DXILPrettyPrinterLegacy, "dxil-pretty-printer",
                      "DXIL Metadata Pretty Printer", true, true)
INITIALIZE_PASS_DEPENDENCY(DXILResourceMDWrapper)
INITIALIZE_PASS_END(DXILPrettyPrinterLegacy, "dxil-pretty-printer",
                    "DXIL Metadata Pretty Printer", true, true)

bool DXILPrettyPrinterLegacy::runOnModule(Module &M) {
  dxil::Resources &Res = getAnalysis<DXILResourceMDWrapper>().getDXILResource();
  prettyPrintResources(OS, Res);
  return false;
}

ModulePass *llvm::createDXILPrettyPrinterLegacyPass(raw_ostream &OS) {
  return new DXILPrettyPrinterLegacy(OS);
}
