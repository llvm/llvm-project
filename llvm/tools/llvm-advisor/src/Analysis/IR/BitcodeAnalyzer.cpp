//===------------------- BitcodeAnalyzer.cpp - LLVM Advisor
//===================//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/IR/BitcodeAnalyzer.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
BitcodeAnalyzer::run(const CapabilityContext &Context) {
  if (Context.IRPath.empty())
    return std::make_unique<JSONCapabilityResult>(
        json::Object{{"capability", getCapabilityID()},
                     {"unit_id", Context.Unit.ID},
                     {"available", false},
                     {"reason", "missing IR artifact"}});

  auto MB = MemoryBuffer::getFile(Context.IRPath);
  if (!MB)
    return createStringError(MB.getError(), "cannot read: %s",
                             Context.IRPath.c_str());

  LLVMContext LLVMCtx;
  SMDiagnostic Err;
  Expected<std::unique_ptr<Module>> MOrErr =
      getLazyBitcodeModule(MB->get()->getMemBufferRef(), LLVMCtx);
  if (!MOrErr)
    return MOrErr.takeError();
  std::unique_ptr<Module> M = std::move(*MOrErr);

  if (M->materializeAll())
    return createStringError(inconvertibleErrorCode(), "cannot materialize: %s",
                             Context.IRPath.c_str());

  return std::make_unique<JSONCapabilityResult>(
      json::Object{{"capability", getCapabilityID()},
                   {"unit_id", Context.Unit.ID},
                   {"module", M->getModuleIdentifier()},
                   {"functions", static_cast<int64_t>(M->size())},
                   {"globals", static_cast<int64_t>(M->global_size())}});
}
