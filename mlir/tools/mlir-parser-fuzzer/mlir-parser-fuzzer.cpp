//===--- mlir-parser-fuzzer.cpp - Entry point to parser fuzzer ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of main so we can build and test without linking libFuzzer.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_USED int LLVMFuzzerTestOneInput(const uint8_t *data,
                                                          size_t size) {
  // Skip empty inputs.
  if (size <= 1 || data[size - 1] != 0)
    return 0;
  --size;

  // Create a null-terminated memory buffer from the input.
  DialectRegistry registry;
  MLIRContext context(registry);
  context.allowUnregisteredDialects();

  // Register diagnostic handler to avoid triggering exit behavior.
  context.getDiagEngine().registerHandler(
      [](mlir::Diagnostic &diag) { return; });

  llvm::StringRef str(reinterpret_cast<const char *>(data), size);

  // Parse module. The parsed module isn't used, so it is discarded post parse
  // (successful or failure). The returned module is wrapped in a unique_ptr
  // such that it is freed upon exit if returned.
  (void)parseSourceString<ModuleOp>(str, &context);
  return 0;
}

extern "C" LLVM_ATTRIBUTE_USED int llvmFuzzerInitialize(int *argc,
                                                        char ***argv) {
  return 0;
}
