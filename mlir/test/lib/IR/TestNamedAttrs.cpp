//===- TestNamedAttrs.cpp - Test passes for MLIR types
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestAttributes.h"
#include "TestDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace test;

namespace {
struct TestNamedAttrsPass
    : public PassWrapper<TestNamedAttrsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestNamedAttrsPass)

  StringRef getArgument() const final { return "test-named-attrs"; }
  StringRef getDescription() const final {
    return "Test support for recursive types";
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto funcName = func.getName();
    // Just make sure recursive types are printed and parsed.
    if (funcName.contains("f_unit_attr")) {
      if (test::TestNamedUnitAttr::has(func)) {
        func.emitRemark() << "found unit attr";
      } else {
        func.emitOpError() << "missing unit attr";
        signalPassFailure();
      }
      return;
    }

    if (funcName.contains("f_int_attr")) {
      if (test::TestNamedIntAttr::has(func)) {
        if (test::TestNamedIntAttr::getValue(func).getInt() == 42) {
          func.emitRemark() << "correct int value";
        } else {
          func.emitOpError() << "wrong int value";
          signalPassFailure();
        }
        return;
      } else {
        func.emitOpError() << "missing int attr";
        signalPassFailure();
      }
      return;
    }

    if (funcName.contains("f_lookup_attr")) {
      func.walk([&](Operation *op) {
        if (test::TestNamedIntAttr::lookupValue(op)) {
          op->emitRemark() << "lookup found attr";
        } else {
          op->emitOpError() << "lookup failed";
          signalPassFailure();
        }
      });
      return;
    }

    if (funcName.contains("f_set_attr")) {
      if (!test::TestNamedIntAttr::has(func)) {
        auto intTy = IntegerType::get(func.getContext(), 32);
        test::TestNamedIntAttr::setValue(func, IntegerAttr::get(intTy, 42));
        func.emitRemark() << "set int attr";
      } else {
        func.emitOpError() << "attr already set";
        signalPassFailure();
      }
      return;
    }

    // Unknown key.
    func.emitOpError() << "unexpected function name";
    signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {

void registerTestNamedAttrsPass() { PassRegistration<TestNamedAttrsPass>(); }

} // namespace test
} // namespace mlir
