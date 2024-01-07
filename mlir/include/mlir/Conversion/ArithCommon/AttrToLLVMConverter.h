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

// Map arithmetic overflow enum values to LLVMIR enum values.
LLVM::IntegerOverflowFlags
convertArithOveflowFlagsToLLVM(arith::IntegerOverflowFlags arithFlags);

// Create an LLVM overflow attribute from a given arithmetic overflow attribute.
LLVM::IntegerOverflowFlagsAttr
convertArithOveflowAttrToLLVM(arith::IntegerOverflowFlagsAttr flagsAttr);

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
    auto arithFMFAttr = dyn_cast_or_null<arith::FastMathFlagsAttr>(
        convertedAttr.erase(arithFMFAttrName));
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

// Attribute converter that populates a NamedAttrList by removing the overflow
// attribute from the source operation attributes, and replacing it with an
// equivalent LLVM overflow attribute.
template <typename SourceOp, typename TargetOp>
class AttrConvertOverflowToLLVM {
public:
  AttrConvertOverflowToLLVM(SourceOp srcOp) {
    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    // Get the name of the arith overflow attribute.
    llvm::StringRef arithAttrName = SourceOp::getIntegerOverflowAttrName();
    // Remove the source overflow attribute.
    auto arithAttr = dyn_cast_or_null<arith::IntegerOverflowFlagsAttr>(
        convertedAttr.erase(arithAttrName));
    if (arithAttr) {
      llvm::StringRef targetAttrName = TargetOp::getIntegerOverflowAttrName();
      convertedAttr.set(targetAttrName,
                        convertArithOveflowAttrToLLVM(arithAttr));
    }
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }

private:
  NamedAttrList convertedAttr;
};
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHCOMMON_ATTRTOLLVMCONVERTER_H
