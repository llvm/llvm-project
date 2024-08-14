//===- CtxProfAnalysis.cpp - contextual profile analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the contextual profile analysis, which maintains contextual
// profiling info through IPO passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "ctx_prof"

using namespace llvm;
cl::opt<std::string>
    UseCtxProfile("use-ctx-profile", cl::init(""), cl::Hidden,
                  cl::desc("Use the specified contextual profile file"));

namespace llvm {
namespace json {
Value toJSON(const PGOCtxProfContext &P) {
  Object Ret;
  Ret["Guid"] = P.guid();
  Ret["Counters"] = Array(P.counters());
  if (P.callsites().empty())
    return Ret;
  auto AllCS =
      ::llvm::map_range(P.callsites(), [](const auto &P) { return P.first; });
  auto MaxIt = ::llvm::max_element(AllCS);
  assert(MaxIt != AllCS.end() && "We should have a max value because the "
                                 "callsites collection is not empty.");
  Array CSites;
  // Iterate to, and including, the maximum index.
  for (auto I = 0U, Max = *MaxIt; I <= Max; ++I) {
    CSites.push_back(Array());
    Array &Targets = *CSites.back().getAsArray();
    if (P.hasCallsite(I))
      for (const auto &[_, Ctx] : P.callsite(I))
        Targets.push_back(toJSON(Ctx));
  }
  Ret["Callsites"] = std::move(CSites);

  return Ret;
}

Value toJSON(const PGOCtxProfContext::CallTargetMapTy &P) {
  Array Ret;
  for (const auto &[_, Ctx] : P)
    Ret.push_back(toJSON(Ctx));
  return Ret;
}
} // namespace json
} // namespace llvm

AnalysisKey CtxProfAnalysis::Key;

CtxProfAnalysis::Result CtxProfAnalysis::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Profile);
  if (auto EC = MB.getError()) {
    M.getContext().emitError("could not open contextual profile file: " +
                             EC.message());
    return {};
  }
  PGOCtxProfileReader Reader(MB.get()->getBuffer());
  auto MaybeCtx = Reader.loadContexts();
  if (!MaybeCtx) {
    M.getContext().emitError("contextual profile file is invalid: " +
                             toString(MaybeCtx.takeError()));
    return {};
  }
  return Result(std::move(*MaybeCtx));
}

PreservedAnalyses CtxProfAnalysisPrinterPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  CtxProfAnalysis::Result &C = MAM.getResult<CtxProfAnalysis>(M);
  if (!C) {
    M.getContext().emitError("Invalid CtxProfAnalysis");
    return PreservedAnalyses::all();
  }
  const auto JSONed = ::llvm::json::toJSON(C.profiles());

  OS << formatv("{0:2}", JSONed);
  OS << "\n";
  return PreservedAnalyses::all();
}
