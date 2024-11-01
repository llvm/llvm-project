//===- AttrToLLVMConverter.h - Arith attributes conversion ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHCOMMON_ATTRTOLLVMCONVERTER_H
#define MLIR_CONVERSION_ARITHCOMMON_ATTRTOLLVMCONVERTER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

//===----------------------------------------------------------------------===//
// Support for converting Arith FastMathFlags to LLVM FastmathFlags
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arith {
// Map arithmetic fastmath enum values to LLVMIR enum values.
LLVM::FastmathFlags
convertArithFastMathFlagsToLLVM(arith::FastMathFlags arithFMF);

// Create an LLVM fastmath attribute from a given arithmetic fastmath attribute.
LLVM::FastmathFlagsAttr
convertArithFastMathAttrToLLVM(arith::FastMathFlagsAttr fmfAttr);

// Attribute converter that populates a NamedAttrList by removing the fastmath
// attribute from the source operation attributes, and replacing it with an
// equivalent LLVM fastmath attribute.
template <typename SourceOp, typename TargetOp>
class AttrConvertFastMathToLLVM {
public:
  AttrConvertFastMathToLLVM(SourceOp srcOp) {
    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    // Get the name of the arith fastmath attribute.
    llvm::StringRef arithFMFAttrName = SourceOp::getFastMathAttrName();
    // Remove the source fastmath attribute.
    auto arithFMFAttr =
        convertedAttr.erase(arithFMFAttrName)
            .template dyn_cast_or_null<arith::FastMathFlagsAttr>();
    if (arithFMFAttr) {
      llvm::StringRef targetAttrName = TargetOp::getFastmathAttrName();
      convertedAttr.set(targetAttrName,
                        convertArithFastMathAttrToLLVM(arithFMFAttr));
    }
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }

private:
  NamedAttrList convertedAttr;
};
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHCOMMON_ATTRTOLLVMCONVERTER_H
