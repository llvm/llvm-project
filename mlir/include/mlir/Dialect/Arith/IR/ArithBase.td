//===- ArithBase.td - Base defs for arith dialect -----------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_BASE
#define ARITH_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Arith_Dialect : Dialect {
  let name = "arith";
  let cppNamespace = "::mlir::arith";
  let description = [{
    The arith dialect is intended to hold basic integer and floating point
    mathematical operations. This includes unary, binary, and ternary arithmetic
    ops, bitwise and shift ops, cast ops, and compare ops. Operations in this
    dialect also accept vectors and tensors of integers or floats. The dialect
    assumes integers are represented by bitvectors with a two's complement
    representation. Unless otherwise stated, the operations within this dialect
    propagate poison values, i.e., if any of its inputs are poison, then the
    output is poison. Unless otherwise stated, operations applied to `vector`
    and `tensor` values propagates poison elementwise.

    Manipulating value with type `i0` isn't supported in this dialect at the
    moment and is considered invalid. This can change in the future if some
    motivating use-cases are presented.

    Some floating-point operations may specify rounding modes and/or fast-math
    flags. In the absence of an explicit rounding mode, the arith dialect uses
    this default round mode for internal purposes such as constant folding and
    canonicalization: round-to-nearest, ties-to-even. The runtime behavior of
    operations without an explicit rounding mode is deferred to the target
    backend and may differ from the default arith rounding mode.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def Arith_CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [
      I64EnumAttrCase<"AlwaysFalse", 0, "false">,
      I64EnumAttrCase<"OEQ", 1, "oeq">,
      I64EnumAttrCase<"OGT", 2, "ogt">,
      I64EnumAttrCase<"OGE", 3, "oge">,
      I64EnumAttrCase<"OLT", 4, "olt">,
      I64EnumAttrCase<"OLE", 5, "ole">,
      I64EnumAttrCase<"ONE", 6, "one">,
      I64EnumAttrCase<"ORD", 7, "ord">,
      I64EnumAttrCase<"UEQ", 8, "ueq">,
      I64EnumAttrCase<"UGT", 9, "ugt">,
      I64EnumAttrCase<"UGE", 10, "uge">,
      I64EnumAttrCase<"ULT", 11, "ult">,
      I64EnumAttrCase<"ULE", 12, "ule">,
      I64EnumAttrCase<"UNE", 13, "une">,
      I64EnumAttrCase<"UNO", 14, "uno">,
      I64EnumAttrCase<"AlwaysTrue", 15, "true">,
    ]> {
  let cppNamespace = "::mlir::arith";
}

def Arith_CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"slt", 2>,
      I64EnumAttrCase<"sle", 3>,
      I64EnumAttrCase<"sgt", 4>,
      I64EnumAttrCase<"sge", 5>,
      I64EnumAttrCase<"ult", 6>,
      I64EnumAttrCase<"ule", 7>,
      I64EnumAttrCase<"ugt", 8>,
      I64EnumAttrCase<"uge", 9>,
    ]> {
  let cppNamespace = "::mlir::arith";
}

def ATOMIC_RMW_KIND_ADDF     : I64EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI     : I64EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ANDI     : I64EnumAttrCase<"andi", 2>;
def ATOMIC_RMW_KIND_ASSIGN   : I64EnumAttrCase<"assign", 3>;
def ATOMIC_RMW_KIND_MAXIMUMF : I64EnumAttrCase<"maximumf", 4>;
def ATOMIC_RMW_KIND_MAXNUMF  : I64EnumAttrCase<"maxnumf", 5>;
def ATOMIC_RMW_KIND_MAXS     : I64EnumAttrCase<"maxs", 6>;
def ATOMIC_RMW_KIND_MAXU     : I64EnumAttrCase<"maxu", 7>;
def ATOMIC_RMW_KIND_MINIMUMF : I64EnumAttrCase<"minimumf", 8>;
def ATOMIC_RMW_KIND_MINNUMF  : I64EnumAttrCase<"minnumf", 9>;
def ATOMIC_RMW_KIND_MINS     : I64EnumAttrCase<"mins", 10>;
def ATOMIC_RMW_KIND_MINU     : I64EnumAttrCase<"minu", 11>;
def ATOMIC_RMW_KIND_MULF     : I64EnumAttrCase<"mulf", 12>;
def ATOMIC_RMW_KIND_MULI     : I64EnumAttrCase<"muli", 13>;
def ATOMIC_RMW_KIND_ORI      : I64EnumAttrCase<"ori", 14>;
def ATOMIC_RMW_KIND_XORI     : I64EnumAttrCase<"xori", 15>;

def AtomicRMWKindAttr : I64EnumAttr<
    "AtomicRMWKind", "",
    [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ANDI,
     ATOMIC_RMW_KIND_ASSIGN, ATOMIC_RMW_KIND_MAXIMUMF, ATOMIC_RMW_KIND_MAXNUMF,
     ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU, ATOMIC_RMW_KIND_MINIMUMF,
     ATOMIC_RMW_KIND_MINNUMF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
     ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI, ATOMIC_RMW_KIND_ORI,
     ATOMIC_RMW_KIND_XORI]> {
  let cppNamespace = "::mlir::arith";
}

def FASTMATH_NONE            : I32BitEnumAttrCaseNone<"none"      >;
def FASTMATH_REASSOC         : I32BitEnumAttrCaseBit<"reassoc",  0>;
def FASTMATH_NO_NANS         : I32BitEnumAttrCaseBit<"nnan",     1>;
def FASTMATH_NO_INFS         : I32BitEnumAttrCaseBit<"ninf",     2>;
def FASTMATH_NO_SIGNED_ZEROS : I32BitEnumAttrCaseBit<"nsz",      3>;
def FASTMATH_ALLOW_RECIP     : I32BitEnumAttrCaseBit<"arcp",     4>;
def FASTMATH_ALLOW_CONTRACT  : I32BitEnumAttrCaseBit<"contract", 5>;
def FASTMATH_APPROX_FUNC     : I32BitEnumAttrCaseBit<"afn",      6>;
def FASTMATH_FAST            : I32BitEnumAttrCaseGroup<
    "fast",
    [
      FASTMATH_REASSOC,         FASTMATH_NO_NANS,     FASTMATH_NO_INFS,
      FASTMATH_NO_SIGNED_ZEROS, FASTMATH_ALLOW_RECIP, FASTMATH_ALLOW_CONTRACT,
      FASTMATH_APPROX_FUNC]>;

def FastMathFlags : I32BitEnumAttr<
    "FastMathFlags",
    "Floating point fast math flags",
    [
      FASTMATH_NONE,           FASTMATH_REASSOC,         FASTMATH_NO_NANS,
      FASTMATH_NO_INFS,        FASTMATH_NO_SIGNED_ZEROS, FASTMATH_ALLOW_RECIP,
      FASTMATH_ALLOW_CONTRACT, FASTMATH_APPROX_FUNC,     FASTMATH_FAST]> {
  let separator = ",";
  let cppNamespace = "::mlir::arith";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def Arith_FastMathAttr :
    EnumAttr<Arith_Dialect, FastMathFlags, "fastmath"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Arith_IntegerOverflowFlags
//===----------------------------------------------------------------------===//

def Arith_IOFnone : I32BitEnumAttrCaseNone<"none">;
def Arith_IOFnsw  : I32BitEnumAttrCaseBit<"nsw", 0>;
def Arith_IOFnuw  : I32BitEnumAttrCaseBit<"nuw", 1>;

def Arith_IntegerOverflowFlags : I32BitEnumAttr<
    "IntegerOverflowFlags",
    "Integer overflow arith flags",
    [Arith_IOFnone, Arith_IOFnsw, Arith_IOFnuw]> {
  let separator = ", ";
  let cppNamespace = "::mlir::arith";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def Arith_IntegerOverflowAttr :
    EnumAttr<Arith_Dialect, Arith_IntegerOverflowFlags, "overflow"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Arith_RoundingMode
//===----------------------------------------------------------------------===//

// These correspond to LLVM's values defined in:
// llvm/include/llvm/ADT/FloatingPointMode.h

def Arith_RToNearestTiesToEven       // Round to nearest, ties to even
    : I32EnumAttrCase<"to_nearest_even", 0>;
def Arith_RDownward                  // Round toward -inf
    : I32EnumAttrCase<"downward", 1>;
def Arith_RUpward                    // Round toward +inf
    : I32EnumAttrCase<"upward", 2>;
def Arith_RTowardZero                // Round toward 0
    : I32EnumAttrCase<"toward_zero", 3>;
def Arith_RToNearestTiesAwayFromZero // Round to nearest, ties away from zero
    : I32EnumAttrCase<"to_nearest_away", 4>;
def Arith_RDynamic                   // Rounding mode unknown at compile time
    : I32EnumAttrCase<"unknown", 7>;  // Mirrors `llvm::RoundingMode::Dynamic`.

def Arith_RoundingModeAttr : I32EnumAttr<
    "RoundingMode", "Floating point rounding mode",
    [Arith_RToNearestTiesToEven, Arith_RDownward, Arith_RUpward,
     Arith_RTowardZero, Arith_RToNearestTiesAwayFromZero, Arith_RDynamic]> {
  let cppNamespace = "::mlir::arith";
}

//===----------------------------------------------------------------------===//
// Floating-point environment enums (used by the `#arith.fenv` attribute)
//===----------------------------------------------------------------------===//

def FPExceptionMode : I32EnumAttr<
    "FPExceptionMode", "floating-point exception mode", [
  I32EnumAttrCase<"Unknown", 0, "unknown">,
  I32EnumAttrCase<"Masked", 1, "masked">,
  I32EnumAttrCase<"Unmasked", 2, "unmasked">
]> {
  let description = [{
    The known floating-point exception mode at the point the instruction is
    executed. If the actual exception mode differs from this value, the
    behavior is undefined.
  }];
  let cppNamespace = "::mlir::arith";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// FenvAttr
//===----------------------------------------------------------------------===//

def Arith_FenvAttr : AttrDef<Arith_Dialect, "Fenv"> {
  let mnemonic = "fenv";
  let summary = "Describes floating-point environment constraints";
  let description = [{
    The `#arith.fenv` attribute describes constraints on the floating-point
    handling of a floating-point operation. It can be attached to
    floating-point operations to capture rounding and exception behavior. All
    of its parameters are optional.

    - `dynamic_rounding_mode`: the known dynamic rounding mode at the point the
      instruction is executed. If the actual dynamic rounding mode when the
      instruction is executed does not match the assumed value, the behavior is
      undefined.
    - `except_mode`: the known exception mode at the point the instruction is
      executed. If the actual state of exception masking when the instruction is
      executed does not match this value, the behavior is undefined.
    - `strict_except`: if `false`, any case that would produce an FP exception
      produces it non-deterministically instead (i.e. it may or may not occur).
      This means the FP status is written non-deterministically, and, if
      exceptions are unmasked, the instruction traps non-deterministically.

    When this attribute is present on an operation, the operation is lowered to
    the corresponding `llvm.experimental.constrained.*` intrinsic during
    translation to LLVM IR.

    The default setting for each parameter (`unknown` for
    `dynamic_rounding_mode`, `masked` for exception_mode, and `false` for
    `strict_except`) is canonically represented as an unset parameter.
    Constructing an attribute whose parameters are all set to their defaults
    therefore yields the same canonical, empty attribute as `#arith.fenv<>`.

    Carrying the (possibly empty) attribute is *not* the same as having no
    attribute at all. An operation that carries the attribute is lowered to the
    corresponding constrained intrinsic operation in LLVM, even when every
    parameter is at its default value. An operation with no `#arith.fenv`
    attribute is lowered to the regular, unconstrained operation.

    Examples:
    ```mlir
    fenv<dynamic_rounding_mode = to_nearest_even>
    fenv<except_mode = unmasked>
    fenv<strict_except = true>
    fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>
    ```
  }];

  let parameters = (ins
    OptionalParameter<
        "std::optional<::mlir::arith::RoundingMode>">:$dynamic_rounding_mode,
    OptionalParameter<"std::optional<::mlir::arith::FPExceptionMode>">:$except_mode,
    OptionalParameter<"mlir::BoolAttr">:$strict_except);

  let assemblyFormat = [{
    `<` struct($dynamic_rounding_mode, $except_mode, $strict_except) `>`
  }];

  // Replace the default builder so that constructing the attribute normalizes
  // any parameter that is set to its default value to "unset". This keeps a
  // single canonical representation for the default floating-point environment.
  // Because `skipDefaultBuilders` suppresses the auto-synthesized const builder
  // call, provide it explicitly. The `DefaultValuedAttr` form below uses a null
  // attribute as its default, so this simply forwards its argument (a null
  // attribute when the operation does not carry a `#arith.fenv` attribute).
  let skipDefaultBuilders = 1;
  let constBuilderCall = "$0";
  let builders = [
    AttrBuilder<(ins
        CArg<"std::optional<::mlir::arith::RoundingMode>",
             "std::nullopt">:$dynamicRoundingMode,
        CArg<"std::optional<::mlir::arith::FPExceptionMode>",
             "std::nullopt">:$exceptMode,
        CArg<"mlir::BoolAttr", "{}">:$strictExcept), [{
      if (dynamicRoundingMode == ::mlir::arith::RoundingMode::unknown)
        dynamicRoundingMode = std::nullopt;
      if (exceptMode == ::mlir::arith::FPExceptionMode::Masked)
        exceptMode = std::nullopt;
      if (strictExcept && !strictExcept.getValue())
        strictExcept = {};
      return Base::get($_ctxt, dynamicRoundingMode, exceptMode, strictExcept);
    }]>
  ];

  let extraClassDeclaration = [{
    /// The default dynamic rounding mode used when `dynamic_rounding_mode` is
    /// unset.
    static constexpr ::mlir::arith::RoundingMode
    getDefaultDynamicRoundingMode() {
      return ::mlir::arith::RoundingMode::unknown;
    }

    /// The default exception mode used when `except_mode` is unset.
    static constexpr ::mlir::arith::FPExceptionMode getDefaultExceptionMode() {
      return ::mlir::arith::FPExceptionMode::Masked;
    }

    /// Returns the effective dynamic rounding mode, substituting the default
    /// when unset.
    ::mlir::arith::RoundingMode getDynamicRoundingModeOrDefault() const {
      return getDynamicRoundingMode().value_or(getDefaultDynamicRoundingMode());
    }
    /// Returns the effective exception mode, substituting the default when
    /// unset.
    ::mlir::arith::FPExceptionMode getExceptionModeOrDefault() const {
      return getExceptMode().value_or(getDefaultExceptionMode());
    }
    /// Returns the effective `strict_except` flag.
    bool getStrictExceptOrDefault() const {
      ::mlir::BoolAttr attr = getStrictExcept();
      return attr ? attr.getValue() : false;
    }

    /// Returns true if every parameter is at its default value, i.e. the
    /// attribute describes the default floating-point environment and imposes
    /// no constraints. Because the builder normalizes defaults to "unset", this
    /// is equivalent to all parameters being unset.
    bool isDefault() const {
      return !getDynamicRoundingMode() && !getExceptMode() && !getStrictExcept();
    }
  }];
}

#endif // ARITH_BASE
