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

/// Maps an arith floating-point exception mode and strict-exception flag, as
/// described by an `#arith.fenv` attribute, to the corresponding LLVM
/// constrained-intrinsic exception behavior:
///
///   masked            + strict_except=false -> fpexcept.ignore
///   masked            + strict_except=true  -> fpexcept.strict
///   unmasked|unknown  + strict_except=false -> fpexcept.maytrap
///   unmasked|unknown  + strict_except=true  -> fpexcept.strict
LLVM::FPExceptionBehavior
convertArithFPExceptionBehaviorToLLVM(arith::FPExceptionMode exceptionMode,
                                      bool strictExcept);

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
  AttrConvertOverflowToLLVM(SourceOp srcOp) {
    using IntegerOverflowFlagsAttr = LLVM::IntegerOverflowFlagsAttr;

    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    // Get the name of the arith overflow attribute.
    StringRef arithAttrName = SourceOp::getIntegerOverflowAttrName();
    // Remove the source overflow attribute from the set that will be present
    // in the target.
    if (auto arithAttr = dyn_cast_if_present<arith::IntegerOverflowFlagsAttr>(
            convertedAttr.erase(arithAttrName))) {
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
  AttrConvertNonNegToLLVM(SourceOp srcOp) {
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    if (!convertedAttr.erase("nonNeg"))
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
  AttrConverterConstrainedFPToLLVM(SourceOp srcOp) {
    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    MLIRContext *ctx = srcOp->getContext();

    // The floating-point environment may be described either by the deprecated
    // `roundingmode` attribute or by the `#arith.fenv` attribute. Collect both
    // (the verifier guarantees they are never set at the same time) and remove
    // them from the attributes carried over to the target op.
    auto roundingModeAttr = dyn_cast_if_present<arith::RoundingModeAttr>(
        convertedAttr.erase(srcOp.getRoundingModeAttrName()));
    arith::FenvAttr fenvAttr = srcOp.getFenvAttr();
    convertedAttr.erase(srcOp.getFenvAttrName());

    // Determine the rounding mode. The `fenv` attribute takes precedence; when
    // neither source carries one, fall back to the default dynamic rounding
    // mode.
    [[maybe_unused]] arith::RoundingMode roundingMode =
        arith::FenvAttr::getDefaultDynamicRoundingMode();
    if (fenvAttr)
      roundingMode = fenvAttr.getDynamicRoundingModeOrDefault();
    else if (roundingModeAttr)
      roundingMode = roundingModeAttr.getValue();

    if constexpr (TargetOp::template hasTrait<
                      LLVM::RoundingModeOpInterface::Trait>()) {
      convertedAttr.set(
          TargetOp::getRoundingModeAttrName(),
          LLVM::RoundingModeAttr::get(
              ctx, convertArithRoundingModeToLLVM(roundingMode)));
    }
    // Constrained intrinsics (llvm.intr.experimental.constrained.*) do not
    // support fastmath flags. Remove the arith fastmath attribute if present.
    if constexpr (SourceOp::template hasTrait<
                      arith::ArithFastMathInterface::Trait>())
      convertedAttr.erase(srcOp.getFastMathAttrName());

    // Determine the exception behavior from the `fenv` attribute, defaulting to
    // `ignore` when no environment is specified (e.g. only a `roundingmode`
    // attribute is present).
    LLVM::FPExceptionBehavior exceptionBehavior =
        LLVM::FPExceptionBehavior::Ignore;
    if (fenvAttr)
      exceptionBehavior = convertArithFPExceptionBehaviorToLLVM(
          fenvAttr.getExceptionModeOrDefault(),
          fenvAttr.getStrictExceptOrDefault());
    convertedAttr.set(
        TargetOp::getFPExceptionBehaviorAttrName(),
        LLVM::FPExceptionBehaviorAttr::get(ctx, exceptionBehavior));
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }
  Attribute getPropAttr() const { return {}; }

private:
  NamedAttrList convertedAttr;
};

} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHCOMMON_ATTRTOLLVMCONVERTER_H
