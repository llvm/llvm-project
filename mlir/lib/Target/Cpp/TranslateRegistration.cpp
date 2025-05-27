//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void registerToCppTranslation() {
  static llvm::cl::opt<bool> declareVariablesAtTop(
      "declare-variables-at-top",
      llvm::cl::desc("Declare variables at top when emitting C/C++"),
      llvm::cl::init(false));

  static llvm::cl::opt<std::string> fileId(
      "file-id", llvm::cl::desc("Emit emitc.file ops with matching id"),
      llvm::cl::init(""));

  static llvm::cl::opt<bool> emitClass(
      "emit-class",
      llvm::cl::desc("If specified, the output will be a class where "
                     "the function(s) in the module are members. "
                     "Enables class-related options."),
      llvm::cl::init(false));

  static llvm::cl::opt<std::string> className(
      "class-name",
      llvm::cl::desc("Mandatory class name if --emit-class is set."),
      llvm::cl::init(""));

  static llvm::cl::opt<std::string> fieldNameAttribute(
      "field-name-attribute",
      llvm::cl::desc("Mandatory name of the attribute to use as field name if "
                     "--emit-class is set."),
      llvm::cl::init(""));

  TranslateFromMLIRRegistration reg(
      "mlir-to-cpp", "translate from mlir to cpp",
      [](Operation *op, raw_ostream &output) {
        if (emitClass) {
          if (className.empty()) {
            llvm::errs() << "Error: --class-name is mandatory when "
                            "--emit-class is set.\n";
            return mlir::failure();
          }
          if (fieldNameAttribute.empty()) {
            llvm::errs() << "Error: --field-name-attribute is mandatory when "
                            "--emit-class is set.\n";
            return mlir::failure();
          }
          return emitc::translateToCpp(
              op, output,
              /*declareVariablesAtTop=*/declareVariablesAtTop,
              /*fileId=*/fileId, /*emitClass=*/emitClass,
              /*className=*/className,
              /*fieldNameAttribute=*/fieldNameAttribute);
        }
        return emitc::translateToCpp(
            op, output,
            /*declareVariablesAtTop=*/declareVariablesAtTop,
            /*fileId=*/fileId, /*emitClass=*/emitClass, /*className=*/className,
            /*fieldNameAttribute=*/fieldNameAttribute);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<cf::ControlFlowDialect,
                        emitc::EmitCDialect,
                        func::FuncDialect>();
        // clang-format on
      });
}

} // namespace mlir
