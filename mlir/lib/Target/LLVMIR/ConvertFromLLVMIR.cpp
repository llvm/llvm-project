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
  static llvm::cl::opt<bool> dropDICompositeTypeElements(
      "drop-di-composite-type-elements",
      llvm::cl::desc(
          "Avoid translating the elements of DICompositeTypes during "
          "the LLVM IR import (discouraged: testing only!)"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> preferUnregisteredIntrinsics(
      "prefer-unregistered-intrinsics",
      llvm::cl::desc(
          "Prefer translating all intrinsics into llvm.call_intrinsic instead "
          "of using dialect supported intrinsics"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> importStructsAsLiterals(
      "import-structs-as-literals",
      llvm::cl::desc("Controls if structs should be imported as literal "
                     "structs, i.e., nameless structs."),
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
          emitError(UnknownLoc::get(context)) << errStr;
          return {};
        }
        if (llvm::verifyModule(*llvmModule, &llvm::errs()))
          return nullptr;

        // Debug records are not currently supported in the LLVM IR translator.
        llvmModule->convertFromNewDbgValues();

        return translateLLVMIRToModule(
            std::move(llvmModule), context, emitExpensiveWarnings,
            dropDICompositeTypeElements, /*loadAllDialects=*/true,
            preferUnregisteredIntrinsics, importStructsAsLiterals);
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
