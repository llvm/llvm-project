//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Translation.h"
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

  TranslateFromMLIRRegistration reg(
      "mlir-to-cpp",
      [](ModuleOp module, raw_ostream &output) {
        return emitc::translateToCpp(
            module, output,
            /*declareVariablesAtTop=*/declareVariablesAtTop);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithmeticDialect,
                        emitc::EmitCDialect,
                        math::MathDialect,
                        StandardOpsDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}

} // namespace mlir
