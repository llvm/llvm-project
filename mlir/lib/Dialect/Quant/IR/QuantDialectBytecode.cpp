//===- QuantDialectBytecode.cpp - Quant Bytecode Implementation
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QuantDialectBytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::quant;

namespace {

static LogicalResult readDoubleAPFloat(DialectBytecodeReader &reader,
                                       double &val) {
  auto valOr =
      reader.readAPFloatWithKnownSemantics(llvm::APFloat::IEEEdouble());
  if (failed(valOr))
    return failure();
  val = valOr->convertToDouble();
  return success();
}

#include "mlir/Dialect/Quant/QuantDialectBytecode.cpp.inc"

/// This class implements the bytecode interface for the Quant dialect.
struct QuantDialectBytecodeInterface : public BytecodeDialectInterface {
  QuantDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};
} // namespace

void quant::detail::addBytecodeInterface(QuantizationDialect *dialect) {
  dialect->addInterfaces<QuantDialectBytecodeInterface>();
}
