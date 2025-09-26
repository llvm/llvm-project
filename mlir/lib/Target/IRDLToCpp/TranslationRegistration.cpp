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

void registerIRDLToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "irdl-to-cpp", "translate IRDL dialect definitions to C++ definitions",
      [](Operation *op, raw_ostream &output) {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case<irdl::DialectOp>([&](irdl::DialectOp dialectOp) {
              return irdl::translateIRDLDialectToCpp(dialectOp, output);
            })
            .Case<ModuleOp>([&](ModuleOp moduleOp) {
              for (Operation &op : moduleOp.getBody()->getOperations())
                if (auto dialectOp = llvm::dyn_cast<irdl::DialectOp>(op))
                  if (failed(
                          irdl::translateIRDLDialectToCpp(dialectOp, output)))
                    return failure();
              return success();
            })
            .Default([](Operation *op) {
              return op->emitError(
                  "unsupported operation for IRDL to C++ translation");
            });
      },
      [](DialectRegistry &registry) { registry.insert<irdl::IRDLDialect>(); });
}

} // namespace mlir
