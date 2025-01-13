//===- TranslationRegistration.cpp - Register translation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/IRDLToCpp/TranslationRegistration.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

using IRDLDialectTranslationFunction =
    const std::function<LogicalResult(irdl::DialectOp, raw_ostream &)>;

static LogicalResult
dispatchTranslation(Operation *op, raw_ostream &output,
                    IRDLDialectTranslationFunction &translation) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<irdl::DialectOp>([&](irdl::DialectOp dialectOp) {
        return translation(dialectOp, output);
      })
      .Case<ModuleOp>([&](ModuleOp moduleOp) {
        for (Operation &op : moduleOp.getBody()->getOperations())
          if (auto dialectOp = llvm::dyn_cast<irdl::DialectOp>(op))
            if (failed(translation(dialectOp, output)))
              return failure();
        return success();
      })
      .Default([](Operation *op) {
        return op->emitError(
            "unsupported operation for IRDL to C++ translation");
      });
}

void registerIRDLToCppTranslation() {
  TranslateFromMLIRRegistration regHeader(
      "irdl-to-cpp-header",
      "translate IRDL dialect definitions to a C++ declaration",
      [](Operation *op, raw_ostream &output) {
        return dispatchTranslation(op, output,
                                   irdl::translateIRDLDialectToCppDeclHeader);
      },
      [](DialectRegistry &registry) { registry.insert<irdl::IRDLDialect>(); });

  TranslateFromMLIRRegistration reg(
      "irdl-to-cpp", "translate IRDL dialect definitions to a C++ definition",
      [](Operation *op, raw_ostream &output) {
        return dispatchTranslation(op, output,
                                   irdl::translateIRDLDialectToCppDef);
      },
      [](DialectRegistry &registry) { registry.insert<irdl::IRDLDialect>(); });
}

} // namespace mlir
