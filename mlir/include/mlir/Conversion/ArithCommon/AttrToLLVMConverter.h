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
/// Maps arithmetic fastmath enum values to LLVM enum values.
LLVM::FastmathFlags
convertArithFastMathFlagsToLLVM(arith::FastMathFlags arithFMF);

/// Creates an LLVM fastmath attribute from a given arithmetic fastmath
/// attribute.
LLVM::FastmathFlagsAttr
convertArithFastMathAttrToLLVM(arith::FastMathFlagsAttr fmfAttr);

/// Maps arithmetic overflow enum values to LLVM enum values.
LLVM::IntegerOverflowFlags
convertArithOverflowFlagsToLLVM(arith::IntegerOverflowFlags arithFlags);

/// Creates an LLVM overflow attribute from a given arithmetic overflow
/// attribute.
LLVM::IntegerOverflowFlagsAttr
convertArithOverflowAttrToLLVM(arith::IntegerOverflowFlagsAttr flagsAttr);

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
    StringRef arithFMFAttrName = SourceOp::getFastMathAttrName();
    // Remove the source fastmath attribute.
    auto arithFMFAttr = dyn_cast_if_present<arith::FastMathFlagsAttr>(
        convertedAttr.erase(arithFMFAttrName));
    if (arithFMFAttr) {
      StringRef targetAttrName = TargetOp::getFastmathAttrName();
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
    StringRef arithAttrName = SourceOp::getIntegerOverflowAttrName();
    // Remove the source overflow attribute.
    auto arithAttr = dyn_cast_if_present<arith::IntegerOverflowFlagsAttr>(
        convertedAttr.erase(arithAttrName));
    if (arithAttr) {
      StringRef targetAttrName = TargetOp::getIntegerOverflowAttrName();
      convertedAttr.set(targetAttrName,
                        convertArithOverflowAttrToLLVM(arithAttr));
    }
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }

private:
  NamedAttrList convertedAttr;
};
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHCOMMON_ATTRTOLLVMCONVERTER_H
