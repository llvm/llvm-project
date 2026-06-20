//===--- IRAnalysisUtils.cpp - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<Module>>
llvm::advisor::parseIRModule(StringRef Path, LLVMContext &Ctx) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(Path, Err, Ctx);
  if (!M)
    return createStringError(inconvertibleErrorCode(),
                             Twine("cannot parse IR: ") + Path);
  return std::move(M);
}

Expected<std::unique_ptr<Module>>
llvm::advisor::parseBitcodeModule(StringRef Path, LLVMContext &Ctx) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createStringError(MB.getError(),
                             Twine("cannot read bitcode: ") + Path);
  Expected<std::unique_ptr<Module>> MOrErr =
      parseBitcodeFile(MB->get()->getMemBufferRef(), Ctx);
  if (!MOrErr)
    return MOrErr.takeError();
  return std::move(*MOrErr);
}
