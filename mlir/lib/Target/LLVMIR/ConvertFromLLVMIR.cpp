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

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

namespace mlir {
void registerFromLLVMIRTranslation() {
  static llvm::cl::opt<bool> emitExpensiveWarnings(
      "emit-expensive-warnings",
      llvm::cl::desc("Emit expensive warnings during LLVM IR import "
                     "(discouraged: testing only!)"),
      llvm::cl::init(false));

  TranslateToMLIRRegistration registration(
      "import-llvm", "Translate LLVMIR to MLIR",
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
        if (llvm::verifyModule(*llvmModule, &llvm::errs()))
          return nullptr;

        return translateLLVMIRToModule(std::move(llvmModule), context,
                                       emitExpensiveWarnings);
      },
      [](DialectRegistry &registry) {
        // Register the DLTI dialect used to express the data layout
        // specification of the imported module.
        registry.insert<DLTIDialect>();
        // Register all dialects that implement the LLVMImportDialectInterface
        // including the LLVM dialect.
        registerAllFromLLVMIRTranslations(registry);
      });
}
} // namespace mlir
