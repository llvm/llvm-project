//===-- ArithOpsInterfaces.td - arith op interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the Arith interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_OPS_INTERFACES
#define ARITH_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def ArithFastMathInterface : OpInterface<"ArithFastMathInterface"> {
  let description = [{
    Access to operation fastmath flags.
  }];

  let cppNamespace = "::mlir::arith";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns a FastMathFlagsAttr attribute for the operation",
      /*returnType=*/  "FastMathFlagsAttr",
      /*methodName=*/  "getFastMathFlagsAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathAttr();
      }]
      >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the FastMathFlagsAttr attribute
                         for the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getFastMathAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "fastmath";
      }]
      >

  ];
}

def ArithIntegerOverflowFlagsInterface : OpInterface<"ArithIntegerOverflowFlagsInterface"> {
  let description = [{
    Access to op integer overflow flags.
  }];

  let cppNamespace = "::mlir::arith";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns an IntegerOverflowFlagsAttr attribute for the operation",
      /*returnType=*/  "IntegerOverflowFlagsAttr",
      /*methodName=*/  "getOverflowAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getOverflowFlagsAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns whether the operation has the No Unsigned Wrap keyword",
      /*returnType=*/  "bool",
      /*methodName=*/  "hasNoUnsignedWrap",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        IntegerOverflowFlags flags = op.getOverflowFlagsAttr().getValue();
        return bitEnumContainsAll(flags, IntegerOverflowFlags::nuw);
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns whether the operation has the No Signed Wrap keyword",
      /*returnType=*/  "bool",
      /*methodName=*/  "hasNoSignedWrap",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        IntegerOverflowFlags flags = op.getOverflowFlagsAttr().getValue();
        return bitEnumContainsAll(flags, IntegerOverflowFlags::nsw);
      }]
      >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the IntegerOverflowFlagsAttr attribute
                         for the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getIntegerOverflowAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "overflowFlags";
      }]
      >
  ];
}

def ArithNonNegFlagInterface : OpInterface<"ArithNonNegFlagInterface"> {
  let description = [{
    Access to op non-negative (nneg) flag.
  }];

  let cppNamespace = "::mlir::arith";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns whether the operation has the nneg flag set",
      /*returnType=*/  "bool",
      /*methodName=*/  "getNonNeg",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getNonNegAttr() != nullptr;
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Set the nneg flag for the operation",
      /*returnType=*/  "void",
      /*methodName=*/  "setNonNeg",
      /*args=*/        (ins "bool":$nonNeg),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        if (nonNeg)
          op.setNonNegAttr(UnitAttr::get(op->getContext()));
        else
          op.removeNonNegAttr();
      }]
      >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the NonNeg flag attribute for
                         the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getNonNegFlagAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "nonNeg";
      }]
      >
  ];
}

def ArithRoundingModeInterface : OpInterface<"ArithRoundingModeInterface"> {
  let description = [{
    Access to op rounding mode.
  }];

  let cppNamespace = "::mlir::arith";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns a RoundingModeAttr attribute for the operation",
      /*returnType=*/  "RoundingModeAttr",
      /*methodName=*/  "getRoundingModeAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getRoundingmodeAttr();
      }]
    >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the RoundingModeAttr attribute for
                         the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getRoundingModeAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "roundingmode";
      }]
    >
  ];
}

def FPEnvConstrainedOpInterface : OpInterface<"FPEnvConstrainedOpInterface"> {
  let description = [{
    This interface is implemented by floating-point operations that can carry an
    optional `#arith.fenv` attribute describing constraints on their
    floating-point environment. It provides a uniform API for querying the
    attribute and its individual fields.

    When the attribute is present, the operation is lowered to the corresponding
    `llvm.experimental.constrained.*` intrinsic during translation to LLVM IR.
  }];

  let cppNamespace = "::mlir::arith";

  let verify = [{
    auto fenvOp = ::llvm::cast<::mlir::arith::FPEnvConstrainedOpInterface>($_op);
    if (fenvOp.getFenvAttr()) {
      if (auto roundingOp =
              ::llvm::dyn_cast<::mlir::arith::ArithRoundingModeInterface>($_op))
        if (roundingOp.getRoundingModeAttr())
          return $_op->emitOpError("The `roundingmode` attribute is deprecated."
                                   " Represent the rounding mode through the "
                                   "`fenv` attribute instead.");
    }
    return ::mlir::success();
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        [{Returns the `#arith.fenv` attribute attached to the
                         operation, or a null attribute if none is present.}],
      /*returnType=*/  "::mlir::arith::FenvAttr",
      /*methodName=*/  "getFenvAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getFenvAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        [{Returns the effective known dynamic rounding mode
                         requested by the attribute, substituting the default
                         (`unknown`) when the attribute is absent or does not set
                         the field.}],
      /*returnType=*/  "::mlir::arith::RoundingMode",
      /*methodName=*/  "getFenvDynamicRoundingMode",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ::mlir::arith::FenvAttr attr = $_op.getFenvAttr();
        return attr ? attr.getDynamicRoundingModeOrDefault()
                    : ::mlir::arith::FenvAttr::getDefaultDynamicRoundingMode();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        [{Returns the effective known exception mode requested by
                         the attribute, substituting the default (`masked`) when
                         the attribute is absent or does not set the field.}],
      /*returnType=*/  "::mlir::arith::FPExceptionMode",
      /*methodName=*/  "getFenvExceptionMode",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ::mlir::arith::FenvAttr attr = $_op.getFenvAttr();
        return attr ? attr.getExceptionModeOrDefault()
                    : ::mlir::arith::FenvAttr::getDefaultExceptionMode();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        [{Returns the effective strict-exception flag requested
                         by the attribute.}],
      /*returnType=*/  "bool",
      /*methodName=*/  "getFenvStrictExcept",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ::mlir::arith::FenvAttr attr = $_op.getFenvAttr();
        return attr ? attr.getStrictExceptOrDefault() : false;
      }]
      >
  ];
}

#endif // ARITH_OPS_INTERFACES
