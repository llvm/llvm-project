//===- ConvertFromLLVMIR.cpp - MLIR to LLVM IR conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the function that registers the translation between
// LLVM IR and the MLIR LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

namespace mlir {
void registerFromLLVMIRTranslation() {
  TranslateToMLIRRegistration registration(
      "import-llvm", "translate mlir to llvmir",
      [](llvm::SourceMgr &sourceMgr,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        llvm::SMDiagnostic err;
        llvm::LLVMContext llvmContext;
        std::unique_ptr<llvm::Module> llvmModule =
            llvm::parseIR(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()),
                          err, llvmContext);
        if (!llvmModule) {
          std::string errStr;
          llvm::raw_string_ostream errStream(errStr);
          err.print(/*ProgName=*/"", errStream);
          emitError(UnknownLoc::get(context)) << errStream.str();
          return {};
        }
        return translateLLVMIRToModule(std::move(llvmModule), context);
      });
}
} // namespace mlir
