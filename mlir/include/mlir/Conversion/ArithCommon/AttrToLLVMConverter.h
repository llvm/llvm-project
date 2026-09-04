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

/// Creates an LLVM rounding mode enum value from a given arithmetic rounding
/// mode enum value.
LLVM::RoundingMode
convertArithRoundingModeToLLVM(arith::RoundingMode roundingMode);

/// Creates an LLVM rounding mode attribute from a given arithmetic rounding
/// mode attribute.
LLVM::RoundingModeAttr
convertArithRoundingModeAttrToLLVM(arith::RoundingModeAttr roundingModeAttr);

/// Returns an attribute for the default LLVM FP exception behavior.
LLVM::FPExceptionBehaviorAttr
getLLVMDefaultFPExceptionBehavior(MLIRContext &context);

// Attribute converter that populates a NamedAttrList by removing the fastmath
// attribute from the source operation attributes, and replacing it with an
// equivalent LLVM fastmath attribute.
template <typename SourceOp, typename TargetOp>
class AttrConvertFastMathToLLVM {
public:
  AttrConvertFastMathToLLVM(SourceOp srcOp)
      : convertedAttr(srcOp->getDiscardableAttrDictionary()) {
    srcOp->getName().populateInherentAttrs(srcOp, convertedAttr);
    convertedAttr.erase(SourceOp::getFastMathAttrName());
    auto arithFMFAttr = srcOp.getFastMathFlagsAttr();
    if (arithFMFAttr) {
      convertedAttr.set(TargetOp::getFastmathAttrName(),
                        convertArithFastMathAttrToLLVM(arithFMFAttr));
    }
  }
  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }
  Attribute getPropAttr() const { return {}; }

private:
  NamedAttrList convertedAttr;
};

// Attribute converter that populates a NamedAttrList by removing the overflow
// attribute from the source operation attributes, and replacing it with an
// equivalent LLVM overflow attribute.
template <typename SourceOp, typename TargetOp>
class AttrConvertOverflowToLLVM {
public:
  AttrConvertOverflowToLLVM(SourceOp srcOp)
      : convertedAttr(srcOp->getDiscardableAttrDictionary()) {
    using IntegerOverflowFlagsAttr = LLVM::IntegerOverflowFlagsAttr;

    if (auto arithAttr = srcOp.getOverflowAttr()) {
      auto llvmFlag = convertArithOverflowFlagsToLLVM(arithAttr.getValue());
      // Create a dictionary attribute holding the overflow flags property.
      // (In the LLVM dialect, the overflow flags are a property, not an
      // attribute.)
      MLIRContext *ctx = srcOp.getOperation()->getContext();
      Builder b(ctx);
      auto llvmFlagAttr = IntegerOverflowFlagsAttr::get(ctx, llvmFlag);
      StringRef llvmAttrName = TargetOp::getOverflowFlagsAttrName();
      NamedAttribute attr{llvmAttrName, llvmFlagAttr};
      // Set the properties attribute of the operation state so that the
      // property can be updated when the operation is created.
      propertiesAttr = b.getDictionaryAttr(ArrayRef(attr));
    }
  }
  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }
  Attribute getPropAttr() const { return propertiesAttr; }

private:
  NamedAttrList convertedAttr;
  DictionaryAttr propertiesAttr;
};

// Attribute converter that populates a NamedAttrList by removing the nonNeg
// attribute from the source operation attributes, and setting it as a property
// on the target LLVM operation.
template <typename SourceOp, typename TargetOp>
class AttrConvertNonNegToLLVM {
public:
  AttrConvertNonNegToLLVM(SourceOp srcOp)
      : convertedAttr(srcOp->getDiscardableAttrDictionary()) {
    if (!srcOp.getNonNeg())
      return;
    MLIRContext *ctx = srcOp.getOperation()->getContext();
    Builder b(ctx);
    NamedAttribute attr{"nonNeg", b.getUnitAttr()};
    propertiesAttr = b.getDictionaryAttr(ArrayRef(attr));
  }
  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }
  Attribute getPropAttr() const { return propertiesAttr; }

private:
  NamedAttrList convertedAttr;
  DictionaryAttr propertiesAttr;
};

template <typename SourceOp, typename TargetOp>
class AttrConverterConstrainedFPToLLVM {
  static_assert(TargetOp::template hasTrait<
                    LLVM::FPExceptionBehaviorOpInterface::Trait>(),
                "Target constrained FP operations must implement "
                "LLVM::FPExceptionBehaviorOpInterface");

public:
  AttrConverterConstrainedFPToLLVM(SourceOp srcOp)
      : convertedAttr(srcOp->getDiscardableAttrDictionary()) {
    if constexpr (TargetOp::template hasTrait<
                      LLVM::RoundingModeOpInterface::Trait>()) {
      auto arithAttr = srcOp.getRoundingModeAttr();
      convertedAttr.set(TargetOp::getRoundingModeAttrName(),
                        convertArithRoundingModeAttrToLLVM(arithAttr));
    }
    // Constrained intrinsics (llvm.intr.experimental.constrained.*) do not
    // support fastmath flags, so do not copy them from the source operation.
    convertedAttr.set(TargetOp::getFPExceptionBehaviorAttrName(),
                      getLLVMDefaultFPExceptionBehavior(*srcOp->getContext()));
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }
  Attribute getPropAttr() const { return {}; }

private:
  NamedAttrList convertedAttr;
};

} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHCOMMON_ATTRTOLLVMCONVERTER_H
