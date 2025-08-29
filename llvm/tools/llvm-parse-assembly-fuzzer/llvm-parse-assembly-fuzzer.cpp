//===-- llvm-parse-assembly-fuzzer.cpp - Fuzz ASM parsing with lib/Fuzzer ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"
#include "llvm/AsmParser/Parser.h"

using namespace llvm;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  LLVMContext Ctx;
  SMDiagnostic Err;

  std::string FuzzInput(reinterpret_cast<const char *>(Data), Size);
  const std::unique_ptr<Module> M =
      parseAssemblyString(FuzzInput.c_str(), Err, Ctx);

  return 0;
}
