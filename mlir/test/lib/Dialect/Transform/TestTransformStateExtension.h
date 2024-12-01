//===- TestTransformStateExtension.h - Test Utility -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an TransformState extension for the purpose of testing the
// relevant APIs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_LIB_DIALECT_TRANSFORM_TESTTRANSFORMSTATEEXTENSION_H
#define MLIR_TEST_LIB_DIALECT_TRANSFORM_TESTTRANSFORMSTATEEXTENSION_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

using namespace mlir;

namespace mlir {
namespace test {
class TestTransformStateExtension
    : public transform::TransformState::Extension {
public:
  TestTransformStateExtension(transform::TransformState &state,
                              StringAttr message)
      : Extension(state), message(message) {}

  StringRef getMessage() const { return message.getValue(); }

  LogicalResult updateMapping(Operation *previous, Operation *updated);

private:
  StringAttr message;
};

class TransformStateInitializerExtension
    : public transform::TransformState::Extension {
public:
  TransformStateInitializerExtension(transform::TransformState &state,
                                     int numOp,
                                     SmallVector<std::string> &registeredOps)
      : Extension(state), numOp(numOp), registeredOps(registeredOps) {}

  int getNumOp() { return numOp; }
  void setNumOp(int num) { numOp = num; }
  SmallVector<std::string> getRegisteredOps() { return registeredOps; }
  void pushRegisteredOps(const std::string &newOp) {
    registeredOps.push_back(newOp);
  }
  std::string printMessage() const {
    std::string message = "Registered transformOps are: ";
    for (const auto &op : registeredOps) {
      message += op + " | ";
    }
    return message;
  }

private:
  int numOp;
  SmallVector<std::string> registeredOps;
};

} // namespace test
} // namespace mlir

#endif // MLIR_TEST_LIB_DIALECT_TRANSFORM_TESTTRANSFORMSTATEEXTENSION_H
