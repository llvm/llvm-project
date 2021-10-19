//===- DecodeAttributesInterfaces.h - DecodeAttributes Interfaces -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_INTERFACES_DECODEATTRIBUTESINTERFACES_H_
#define MLIR_INTERFACES_DECODEATTRIBUTESINTERFACES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// Define an interface to decode opaque constant tensor.
class DialectDecodeAttributesInterface
    : public DialectInterface::Base<DialectDecodeAttributesInterface> {
public:
  DialectDecodeAttributesInterface(Dialect *dialect) : Base(dialect) {}

  /// Registered hook to decode opaque constants associated with this
  /// dialect. The hook function attempts to decode an opaque constant tensor
  /// into a tensor with non-opaque content. If decoding is successful, this
  /// method returns success() and sets 'output' attribute. If not, it returns
  /// failure() and leaves 'output' unspecified. The default hook fails to
  /// decode.
  virtual LogicalResult decode(OpaqueElementsAttr input,
                               ElementsAttr &output) const {
    return failure();
  }
};

} // end namespace mlir

#endif // MLIR_INTERFACES_DECODEATTRIBUTESINTERFACES_H_
