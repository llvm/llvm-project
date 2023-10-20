//===-- CommonTypeConstraints.td - Common Type Constraints--*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains commonly used type constraints.
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_TYPE_CONSTRAINTS_TD
#define COMMON_TYPE_CONSTRAINTS_TD

include "mlir/IR/Constraints.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Common predicates
//===----------------------------------------------------------------------===//

// Whether a type is a VectorType.
// Explicitly disallow 0-D vectors for now until we have good enough coverage.
def IsVectorTypePred : And<[CPred<"::llvm::isa<::mlir::VectorType>($_self)">,
                            CPred<"::llvm::cast<::mlir::VectorType>($_self).getRank() > 0">]>;

// Temporary vector type clone that allows gradual transition to 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
def IsVectorOfAnyRankTypePred : CPred<"::llvm::isa<::mlir::VectorType>($_self)">;

// Whether a type is a fixed-length VectorType.
def IsFixedVectorTypePred : CPred<[{::llvm::isa<::mlir::VectorType>($_self) &&
                                  !::llvm::cast<VectorType>($_self).isScalable()}]>;

// Whether a type is a scalable VectorType.
def IsVectorTypeWithAnyDimScalablePred 
        : CPred<[{::llvm::isa<::mlir::VectorType>($_self) &&
                  ::llvm::cast<VectorType>($_self).isScalable()}]>;

// Whether a type is a scalable VectorType, with a single trailing scalable dimension.
// Examples:
// Valid:
//   - vector<[4]xf32>, vector<2x3x[2]xi64>, vector<32x[8]xi32>
// Invalid
//   - vector<[4]x8xi32>, vector<[2]x[2]xf64>, vector<2x[8]x4xi32>
def IsVectorTypeWithOnlyTrailingDimScalablePred : And<[
  CPred<"::llvm::isa<::mlir::VectorType>($_self)">,
  CPred<"::llvm::cast<::mlir::VectorType>($_self).getRank() > 0">,
  CPred<"::llvm::cast<::mlir::VectorType>($_self).getScalableDims().back()">,
  CPred<"!llvm::is_contained(::llvm::cast<::mlir::VectorType>($_self).getScalableDims().drop_back(), true)">
]>;

// Whether a type is a VectorType and all dimensions are scalable.
def IsVectorTypeWithAllDimsScalablePred : And<[
  IsVectorTypePred,
  CPred<[{::llvm::cast<::mlir::VectorType>($_self).allDimsScalable()}]>
]>;

// Whether a type is a TensorType.
def IsTensorTypePred : CPred<"::llvm::isa<::mlir::TensorType>($_self)">;

// Whether a type is a MemRefType.
def IsMemRefTypePred : CPred<"::llvm::isa<::mlir::MemRefType>($_self)">;

// Whether a type is an UnrankedMemRefType
def IsUnrankedMemRefTypePred
        : CPred<"::llvm::isa<::mlir::UnrankedMemRefType>($_self)">;

// Whether a type is an UnrankedTensorType
def IsUnrankedTensorTypePred
        : CPred<"::llvm::isa<::mlir::UnrankedTensorType>($_self)">;

// Whether a type is a RankedTensorType
def IsRankedTensorTypePred
        : CPred<"::llvm::isa<::mlir::RankedTensorType>($_self)">;

// Whether a type is a BaseMemRefType
def IsBaseMemRefTypePred
        : CPred<"::llvm::isa<::mlir::BaseMemRefType>($_self)">;

// Whether a type is a ShapedType.
def IsShapedTypePred : CPred<"::llvm::isa<::mlir::ShapedType>($_self)">;

// For a ShapedType, verify that it has a static shape.
def HasStaticShapePred :
        CPred<"::llvm::cast<::mlir::ShapedType>($_self).hasStaticShape()">;

// Whether a type is a TupleType.
def IsTupleTypePred : CPred<"::llvm::isa<::mlir::TupleType>($_self)">;

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

// A type, carries type constraints.
class Type<Pred condition, string descr = "",
           string cppClassName = "::mlir::Type"> :
    TypeConstraint<condition, descr, cppClassName> {
  string description = "";
  string builderCall = "";
}

// Allows providing an alternative name and summary to an existing type def.
class TypeAlias<Type t, string summary = t.summary> :
    Type<t.predicate, summary, t.cppClassName> {
  let description = t.description;
  let builderCall = t.builderCall;
}

// A type of a specific dialect.
class DialectType<Dialect d, Pred condition, string descr = "",
                  string cppClassName = "::mlir::Type"> :
    Type<condition, descr, cppClassName> {
  Dialect dialect = d;
}

// A variadic type constraint. It expands to zero or more of the base type. This
// class is used for supporting variadic operands/results.
class Variadic<Type type> : TypeConstraint<type.predicate,
                                           "variadic of " # type.summary,
                                           type.cppClassName> {
  Type baseType = type;
  int minSize = 0;
}

// A nested variadic type constraint. It expands to zero or more variadic ranges
// of the base type. This class is used for supporting variadic operands and
// results. `variadicSegmentAttrName` should correspond to the name of an
// DenseI32ArrayAttr argument that provides the sizes of the inner variadic
// operand groups.
class VariadicOfVariadic<Type type, string variadicSegmentAttrName>
    : Variadic<type> {
  string segmentAttrName = variadicSegmentAttrName;
}

// An optional type constraint. It expands to either zero or one of the base
// type. This class is used for supporting optional operands/results.
class Optional<Type type> : TypeConstraint<type.predicate, type.summary,
                                           type.cppClassName> {
  Type baseType = type;
}

// A type that can be constructed using MLIR::Builder.
// Note that this does not "inherit" from Type because it would require
// duplicating Type subclasses for buildable and non-buildable cases to avoid
// diamond "inheritance".
// TODO: we may extend this to a more general 'Buildable' trait, making some
// Types and some Attrs buildable.
class BuildableType<code builder> {
  // The builder call to invoke (if specified) to construct the BuildableType.
  code builderCall = builder;
}

// A type that's buildable iff the type passed as an argument is buildable.
// This is intended for use by types like container types, which are only
// buildable if the type of their elements is buildable.
class SameBuildabilityAs<Type type, code builder> {
  code builderCall = !if(!empty(type.builderCall), "", builder);
}

// Any type at all.
def AnyType : Type<CPred<"true">, "any type">;

// None type
def NoneType : Type<CPred<"::llvm::isa<::mlir::NoneType>($_self)">, "none type",
                    "::mlir::NoneType">,
      BuildableType<"$_builder.getType<::mlir::NoneType>()">;

// Any type from the given list
class AnyTypeOf<list<Type> allowedTypes, string summary = "",
                string cppClassName = "::mlir::Type"> : Type<
    // Satisfy any of the allowed types' conditions.
    Or<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypes, t.summary), " or "),
        summary),
    cppClassName>;

// A type that satisfies the constraints of all given types.
class AllOfType<list<Type> allowedTypes, string summary = "",
                string cppClassName = "::mlir::Type"> : Type<
    // Satisfy all of the allowedf types' conditions.
    And<!foreach(allowedType, allowedTypes, allowedType.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypes, t.summary), " and "),
        summary),
    cppClassName>;

// A type that satisfies additional predicates.
class ConfinedType<Type type, list<Pred> predicates, string summary = "",
                   string cppClassName = type.cppClassName> : Type<
    And<!listconcat([type.predicate], !foreach(pred, predicates, pred))>,
    summary, cppClassName>;

// Integer types.

// Any integer type irrespective of its width and signedness semantics.
def AnyInteger : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self)">, "integer",
                      "::mlir::IntegerType">;

// Any integer type (regardless of signedness semantics) of a specific width.
class AnyI<int width>
    : Type<CPred<"$_self.isInteger(" # width # ")">, width # "-bit integer"> {
  int bitwidth = width;
}

class AnyIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, AnyI<w>),
              !interleave(widths, "/") # "-bit integer",
              "::mlir::IntegerType">;

def AnyI1  : AnyI<1>;
def AnyI8  : AnyI<8>;
def AnyI16 : AnyI<16>;
def AnyI32 : AnyI<32>;
def AnyI64 : AnyI<64>;

// Any signless integer type irrespective of its width.
def AnySignlessInteger : Type<
  CPred<"$_self.isSignlessInteger()">, "signless integer",
        "::mlir::IntegerType">;

// Signless integer type of a specific width.
class I<int width>
    : Type<CPred<"$_self.isSignlessInteger(" # width # ")">,
                  width # "-bit signless integer", "::mlir::IntegerType">,
      BuildableType<"$_builder.getIntegerType(" # width # ")"> {
  int bitwidth = width;
}

class SignlessIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, I<w>),
              !interleave(widths, "/") # "-bit signless integer">;

def I1  : I<1>;
def I8  : I<8>;
def I16 : I<16>;
def I32 : I<32>;
def I64 : I<64>;
def I128 : I<128>;

// Any signed integer type irrespective of its width.
def AnySignedInteger : Type<
  CPred<"$_self.isSignedInteger()">, "signed integer">;

// Signed integer type of a specific width.
class SI<int width>
    : Type<CPred<"$_self.isSignedInteger(" # width # ")">,
                  width # "-bit signed integer", "::mlir::IntegerType">,
      BuildableType<
        "$_builder.getIntegerType(" # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

class SignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, SI<w>),
              !interleave(widths, "/") # "-bit signed integer">;

def SI1  : SI<1>;
def SI8  : SI<8>;
def SI16 : SI<16>;
def SI32 : SI<32>;
def SI64 : SI<64>;

// Any unsigned integer type irrespective of its width.
def AnyUnsignedInteger : Type<
  CPred<"$_self.isUnsignedInteger()">, "unsigned integer">;

// Unsigned integer type of a specific width.
class UI<int width>
    : Type<CPred<"$_self.isUnsignedInteger(" # width # ")">,
                  width # "-bit unsigned integer", "::mlir::IntegerType">,
      BuildableType<
        "$_builder.getIntegerType(" # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

class UnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, UI<w>),
              !interleave(widths, "/") # "-bit unsigned integer">;

def UI1  : UI<1>;
def UI8  : UI<8>;
def UI16 : UI<16>;
def UI32 : UI<32>;
def UI64 : UI<64>;

// Index type.
def Index : Type<CPred<"::llvm::isa<::mlir::IndexType>($_self)">, "index",
                 "::mlir::IndexType">,
            BuildableType<"$_builder.getIndexType()">;

// Any signless integer type or index type.
def AnySignlessIntegerOrIndex : Type<CPred<"$_self.isSignlessIntOrIndex()">,
                                     "signless integer or index">;

// Floating point types.

// Any float type irrespective of its width.
def AnyFloat : Type<CPred<"::llvm::isa<::mlir::FloatType>($_self)">, "floating-point",
                    "::mlir::FloatType">;

// Float type of a specific width.
class F<int width>
    : Type<CPred<"$_self.isF" # width # "()">,
           width # "-bit float", "::mlir::FloatType">,
      BuildableType<"$_builder.getF" # width # "Type()"> {
  int bitwidth = width;
}

class FloatOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, F<w>),
              !interleave(widths, "/") # "-bit float">;

def F16 : F<16>;
def F32 : F<32>;
def F64 : F<64>;
def F80 : F<80>;
def F128 : F<128>;

def BF16 : Type<CPred<"$_self.isBF16()">, "bfloat16 type">,
           BuildableType<"$_builder.getBF16Type()">;
def TF32 : Type<CPred<"$_self.isTF32()">, "tf32 type">,
           BuildableType<"$_builder.getTF32Type()">;
def F8E4M3FN : Type<CPred<"$_self.isFloat8E4M3FN()">, "f8E4M3FN type">,
               BuildableType<"$_builder.getFloat8E4M3FNType()">;
def F8E5M2 : Type<CPred<"$_self.isFloat8E5M2()">, "f8E5M2 type">,
             BuildableType<"$_builder.getFloat8E5M2Type()">;
def F8E4M3FNUZ : Type<CPred<"$_self.isFloat8E4M3FNUZ()">, "f8E4M3FNUZ type">,
                 BuildableType<"$_builder.getFloat8E4M3FNUZType()">;
def F8E4M3B11FNUZ : Type<CPred<"$_self.isFloat8E4M3B11FNUZ()">, "f8E4M3B11FNUZ type">,
                 BuildableType<"$_builder.getFloat8E4M3B11FNUZType()">;
def F8E5M2FNUZ : Type<CPred<"$_self.isFloat8E5M2FNUZ()">, "f8E5M2FNUZ type">,
                 BuildableType<"$_builder.getFloat8E5M2FNUZType()">;

def AnyComplex : Type<CPred<"::llvm::isa<::mlir::ComplexType>($_self)">,
                      "complex-type", "::mlir::ComplexType">;

class Complex<Type type>
    : ConfinedType<AnyComplex, [
          SubstLeaves<"$_self",
                      "::llvm::cast<::mlir::ComplexType>($_self).getElementType()",
           type.predicate>],
           "complex type with " # type.summary # " elements",
           "::mlir::ComplexType">,
      SameBuildabilityAs<type, "::mlir::ComplexType::get($_builder.get" # type #
                               "Type())"> {
  Type elementType = type;
}

class OpaqueType<string dialect, string name, string summary>
  : Type<CPred<"isOpaqueTypeWithName($_self, \""#dialect#"\", \""#name#"\")">,
         summary, "::mlir::OpaqueType">,
    BuildableType<"::mlir::OpaqueType::get("
                  "$_builder.getStringAttr(\"" # dialect # "\"), \""
                  # name # "\")">;

// Function Type

// Any function type.
def FunctionType : Type<CPred<"::llvm::isa<::mlir::FunctionType>($_self)">,
                              "function type", "::mlir::FunctionType">;

// A container type is a type that has another type embedded within it.
class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
                    string descr, string cppClassName = "::mlir::Type"> :
    // First, check the container predicate.  Then, substitute the extracted
    // element into the element type checker.
    Type<And<[containerPred,
                SubstLeaves<"$_self", !cast<string>(elementTypeCall),
                etype.predicate>]>,
         descr # " of " # etype.summary # " values", cppClassName>;

class ShapedContainerType<list<Type> allowedTypes,
                          Pred containerPred, string descr,
                          string cppClassName = "::mlir::Type"> :
    Type<And<[containerPred,
              Concat<"[](::mlir::Type elementType) { return ",
                SubstLeaves<"$_self", "elementType",
                AnyTypeOf<allowedTypes>.predicate>,
                "; }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())">]>,
         descr # " of " # AnyTypeOf<allowedTypes>.summary # " values", cppClassName>;

// Whether a shaped type is ranked.
def HasRankPred : CPred<"::llvm::cast<::mlir::ShapedType>($_self).hasRank()">;

// Whether a shaped type has one of the specified ranks.
class HasAnyRankOfPred<list<int> ranks> : And<[
    HasRankPred,
    Or<!foreach(rank, ranks,
                CPred<[{::llvm::cast<::mlir::ShapedType>($_self).getRank()
                         == }]
                      # rank>)>]>;

// Whether a shaped type has a rank greater than or equal of the specified rank.
class HasRankGreaterOrEqualPred<int rank> : And<[
    HasRankPred,
    CPred<[{::llvm::cast<::mlir::ShapedType>($_self).getRank() >= }] # rank>
]>;

// Vector types.

class VectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypePred, "vector",
                      "::mlir::VectorType">;

// Temporary vector type clone that allows gradual transition to 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
class VectorOfAnyRankOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorOfAnyRankTypePred, "vector",
                      "::mlir::VectorType">;

class FixedVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsFixedVectorTypePred,
          "fixed-length vector", "::mlir::VectorType">;

class ScalableVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypeWithAnyDimScalablePred,
          "scalable vector", "::mlir::VectorType">;

// Any vector with a single trailing scalable dimension, with an element type in
// the `allowedTypes` list.
//
// Note: This Similar to ScalableVectorOf, with the extra requirement that only
// the trailing dim is scalable.
class VectorWithTrailingDimScalableOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypeWithOnlyTrailingDimScalablePred,
          "trailing scalable vector", "::mlir::VectorType">;

// Whether the number of elements of a vector is from the given
// `allowedRanks` list
class IsVectorOfRankPred<list<int> allowedRanks> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a fixed-length vector is from the given
// `allowedRanks` list
class IsFixedVectorOfRankPred<list<int> allowedRanks> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a scalable vector is from the given
// `allowedRanks` list
class IsScalableVectorOfRankPred<list<int> allowedRanks> :
  And<[IsVectorTypeWithAnyDimScalablePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// Any vector where the rank is from the given `allowedRanks` list
class VectorOfRank<list<int> allowedRanks> : Type<
  IsVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// Any fixed-length vector where the rank is from the given `allowedRanks` list
class FixedVectorOfRank<list<int> allowedRanks> : Type<
  IsFixedVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// Any scalable vector where the rank is from the given `allowedRanks` list
class ScalableVectorOfRank<list<int> allowedRanks> : Type<
  IsScalableVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// Any vector where the rank is from the given `allowedRanks` list and the type
// is from the given `allowedTypes` list
class VectorOfRankAndType<list<int> allowedRanks,
                          list<Type> allowedTypes> : AllOfType<
  [VectorOf<allowedTypes>, VectorOfRank<allowedRanks>],
  VectorOf<allowedTypes>.summary # VectorOfRank<allowedRanks>.summary,
  "::mlir::VectorType">;

// Whether the number of elements of a vector is from the given
// `allowedLengths` list
class IsVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a fixed-length vector is from the given
// `allowedLengths` list
class IsFixedVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a scalable vector is from the given
// `allowedLengths` list
class IsScalableVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsVectorTypeWithAnyDimScalablePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Normalizes an index so the indices in both directions have the same value.
// For example, when indexing forwards index 2 is the third element. When
// indexing in reverse the third element is -3. This helper would map both of
// these to the "normalized" index of 3. This makes the bounds checking in
// IsNthDimSizeIsOneOfPred simpler (see first CPred).
class NormalizeIndex<int value> {
  int ret = !if(!lt(value, 0),
    !sub(0, value)  /* -value if negative */,
    !add(value, 1)  /* value + 1 if positive*/);
}

// Whether the n-th dim of the shape is contained within `allowedSizes`.
// Negative values for `n` index in reverse.
//
// Examples:
// IsNthDimSizeIsOneOfPred<0, {2, 3, 4}>
//  - Accepts any shape where the first dim is 2, 3, or 4.
//    * This means shapes like: 2x8x9x5, 4, 3x1, 4x?, etc
// IsNthDimSizeIsOneOfPred<-1, {16}>
//  - Accepts any shape where the last dim is 16.
//    * This means shapes like 2x16, 16, 1x2x3x4x16, etc
// IsNthDimSizeIsOneOfPred<-2, {10, 5}>
//  - Accepts any shape where the second to last dim is 10 or 5.
//    * This means shapes like: 1x10x2, 2x1x4x5x6, 8x10x?, etc
class IsNthDimSizeIsOneOfPred<int n, list<int> allowedSizes>
  : And<[
      CPred<"::llvm::cast<::mlir::ShapedType>($_self).getRank() >= " # NormalizeIndex<n>.ret>,
      CPred<"::llvm::is_contained(ArrayRef<int64_t>({" # !interleave(allowedSizes, ", ") # "}), "
        # "::llvm::cast<::mlir::ShapedType>($_self).getDimSize("
        #   !if(!lt(n, 0),
              "::llvm::cast<::mlir::ShapedType>($_self).getRank() + " # n,
              "" # n)
        # "))">]>;

// Whether the shape of a vector matches the given `shape` list.
class IsVectorOfShape<list<int> shape>
  : CPred<"::llvm::cast<::mlir::VectorType>($_self).getShape() == ArrayRef<int64_t>({" # !interleave(shape, ", ") # "})">;

// Any vector where the number of elements is from the given
// `allowedLengths` list
class VectorOfLength<list<int> allowedLengths> : Type<
  IsVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// Any fixed-length vector where the number of elements is from the given
// `allowedLengths` list
class FixedVectorOfLength<list<int> allowedLengths> : Type<
  IsFixedVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// Any scalable vector where the number of elements is from the given
// `allowedLengths` list
class ScalableVectorOfLength<list<int> allowedLengths> : Type<
  IsScalableVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// Any vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes`
// list
class VectorOfLengthAndType<list<int> allowedLengths,
                            list<Type> allowedTypes> : AllOfType<
  [VectorOf<allowedTypes>, VectorOfLength<allowedLengths>],
  VectorOf<allowedTypes>.summary # VectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// Any fixed-length vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes` list
class FixedVectorOfLengthAndType<list<int> allowedLengths,
                                    list<Type> allowedTypes> : AllOfType<
  [FixedVectorOf<allowedTypes>, FixedVectorOfLength<allowedLengths>],
  FixedVectorOf<allowedTypes>.summary #
  FixedVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// Any scalable vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes` list
class ScalableVectorOfLengthAndType<list<int> allowedLengths,
                                    list<Type> allowedTypes> : AllOfType<
  [ScalableVectorOf<allowedTypes>, ScalableVectorOfLength<allowedLengths>],
  ScalableVectorOf<allowedTypes>.summary #
  ScalableVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// Any scalable vector where the rank is from the given `allowedRanks` list and
// the number of elements is from the given `allowedLengths` list and the type
// is from the given `allowedTypes` list
class ScalableVectorOfRankAndLengthAndType<list<int> allowedRanks,
                                           list<int> allowedLengths,
                                           list<Type> allowedTypes> : AllOfType<
  [ScalableVectorOfRank<allowedRanks>, ScalableVectorOf<allowedTypes>,
   ScalableVectorOfLength<allowedLengths>],
  ScalableVectorOfRank<allowedRanks>.summary #
  ScalableVectorOf<allowedTypes>.summary #
  ScalableVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// Any ShapedType where the size of the n-th dim is contained in `allowedSizes`.
// Negative values for `n` index in reverse.
class ShapedTypeWithNthDimOfSize<int n, list<int> allowedSizes> : Type<
  IsNthDimSizeIsOneOfPred<n, allowedSizes>,
  " with dim " # n # " having a size of {" # !interleave(allowedSizes, ", ") # "}",
  "::mlir::ShapedType">;

// Any scalable vector with a single trailing scalable dimensions, where the
// size of the trailing dimension is in `allowedTrailingSizes` list, and the
// type is in the `allowedTypes` list.
class VectorWithTrailingDimScalableOfSizeAndType<list<int> allowedTrailingSizes,
                                           list<Type> allowedTypes> : AllOfType<
  [VectorWithTrailingDimScalableOf<allowedTypes>,
   ShapedTypeWithNthDimOfSize<-1, allowedTrailingSizes>],
   VectorWithTrailingDimScalableOf<allowedTypes>.summary #
   ShapedTypeWithNthDimOfSize<-1, allowedTrailingSizes>.summary,
  "::mlir::VectorType">;

def AnyVector : VectorOf<[AnyType]>;
// Temporary vector type clone that allows gradual transition to 0-D vectors.
def AnyVectorOfAnyRank : VectorOfAnyRankOf<[AnyType]>;

def AnyFixedVector : FixedVectorOf<[AnyType]>;

def AnyScalableVector : ScalableVectorOf<[AnyType]>;

// Shaped types.

def AnyShaped: ShapedContainerType<[AnyType], IsShapedTypePred, "shaped",
                                   "::mlir::ShapedType">;

//===----------------------------------------------------------------------===//
// Tensor types.

// Unranked tensor type whose element type is from the given `allowedTypes`
// list, and which additionally satisfies an optional list of predicates.
class UnrankedTensorOf<list<Type> allowedTypes, list<Pred> preds = [],
                       string summary = "unranked tensor">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([IsUnrankedTensorTypePred], preds)>,
      summary, "::mlir::UnrankedTensorType">;

// Ranked tensor type whose element type is from the given `allowedTypes` list,
// and which additionally satisfies an optional list of predicates.
class RankedTensorOf<list<Type> allowedTypes, list<Pred> preds = [],
                     string summary = "ranked tensor">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([IsRankedTensorTypePred], preds)>,
      summary, "::mlir::RankedTensorType">;

// Any tensor type whose element type is from the given `allowedTypes`
// list, and which additionally satisfies an optional list of predicates.
//
// TODO: use `Constraint` instead of `Pred`, so we can generate a better
// default summary (a la `ConfinedAttr`).
class TensorOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "tensor">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsTensorTypePred], preds)>,
      summary, "::mlir::TensorType">;

def AnyTensor  : TensorOf<[AnyType]>;

def I1Tensor   : TensorOf<[I1]>;
def I8Tensor   : TensorOf<[I8]>;
def I16Tensor  : TensorOf<[I16]>;
def I32Tensor  : TensorOf<[I32]>;
def I64Tensor  : TensorOf<[I64]>;
def IndexTensor: TensorOf<[Index]>;

def BF16Tensor : TensorOf<[BF16]>;
def F16Tensor  : TensorOf<[F16]>;
def F32Tensor  : TensorOf<[F32]>;
def F64Tensor  : TensorOf<[F64]>;

class Non0RankedTensorOf<list<Type> allowedTypes>
  : TensorOf<allowedTypes, [HasRankGreaterOrEqualPred<1>],
      "non-0-ranked.tensor">;

def AnyRankedTensor : RankedTensorOf<[AnyType]>;
def AnyNon0RankedTensor  : Non0RankedTensorOf<[AnyType]>;
def AnyUnrankedTensor  : UnrankedTensorOf<[AnyType]>;

def AnyNon0RankedOrUnrankedTensor
  : AnyTypeOf<[AnyUnrankedTensor, AnyNon0RankedTensor],
              "non-0-ranked or unranked tensor", "::mlir::TensorType">;

// Ranked tensor type with one of the specified types and ranks.
class TensorRankOf<list<Type> allowedTypes, list<int> ranks>
  : RankedTensorOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tensor">;

class 0DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [0]>;
class 1DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [1]>;
class 2DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [2]>;
class 3DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [3]>;
class 4DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [4]>;

class StaticShapeTensorOf<list<Type> allowedTypes>
  : RankedTensorOf<allowedTypes, [HasStaticShapePred],
                   "statically shaped tensor">;

def AnyStaticShapeTensor : StaticShapeTensorOf<[AnyType]>;

//===----------------------------------------------------------------------===//
// Memref type.

// Any unranked memref whose element type is from the given `allowedTypes` list.
class UnrankedMemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes,
                        IsUnrankedMemRefTypePred, "unranked.memref",
                        "::mlir::UnrankedMemRefType">;

def AnyUnrankedMemRef : UnrankedMemRefOf<[AnyType]>;

// Any ranked memref whose element type is from the given `allowedTypes` list.
class MemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsMemRefTypePred, "memref",
                        "::mlir::MemRefType">;

class Non0RankedMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasRankGreaterOrEqualPred<1>],
         "non-0-ranked." # MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

def AnyMemRef : MemRefOf<[AnyType]>;
def AnyNon0RankedMemRef : Non0RankedMemRefOf<[AnyType]>;

// Any memref (ranked or unranked) whose element type is from the given
// `allowedTypes` list, and which additionally satisfies an optional list of
// predicates.
class RankedOrUnrankedMemRefOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "ranked or unranked memref">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsBaseMemRefTypePred], preds)>,
      summary, "::mlir::BaseMemRefType">;

def AnyRankedOrUnrankedMemRef  : RankedOrUnrankedMemRefOf<[AnyType]>;
def AnyNon0RankedOrUnrankedMemRef:
    AnyTypeOf<[AnyUnrankedMemRef, AnyNon0RankedMemRef]>;

// Memref declarations handle any memref, independent of rank, size, (static or
// dynamic), layout, or memory space.
def I1MemRef  : MemRefOf<[I1]>;
def I8MemRef  : MemRefOf<[I8]>;
def I16MemRef : MemRefOf<[I16]>;
def I32MemRef : MemRefOf<[I32]>;
def I64MemRef : MemRefOf<[I64]>;

def BF16MemRef : MemRefOf<[BF16]>;
def F16MemRef  : MemRefOf<[F16]>;
def F32MemRef  : MemRefOf<[F32]>;
def F64MemRef  : MemRefOf<[F64]>;

// TODO: Have an easy way to add another constraint to a type.
class MemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    ConfinedType<MemRefOf<allowedTypes>, [HasAnyRankOfPred<ranks>],
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

class StaticShapeMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasStaticShapePred],
         "statically shaped " # MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

def AnyStaticShapeMemRef : StaticShapeMemRefOf<[AnyType]>;

// For a MemRefType, verify that it has strides.
def HasStridesPred : CPred<[{ isStrided(::llvm::cast<::mlir::MemRefType>($_self)) }]>;

class StridedMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasStridesPred],
         "strided " # MemRefOf<allowedTypes>.summary>;

def AnyStridedMemRef : StridedMemRefOf<[AnyType]>;

class AnyStridedMemRefOfRank<int rank> :
  AllOfType<[AnyStridedMemRef, MemRefRankOf<[AnyType], [rank]>],
       AnyStridedMemRef.summary # " of rank " # rank>;

class StridedMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    ConfinedType<MemRefOf<allowedTypes>, [HasAnyRankOfPred<ranks>],
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary>;

// This represents a generic tuple without any constraints on element type.
def AnyTuple : Type<IsTupleTypePred, "tuple", "::mlir::TupleType">;

// A container type that has other types embedded in it, but (unlike
// ContainerType) can hold elements with a mix of types. Requires a call that
// produces a list of all elements' types.
class MixedContainerType<Type etype, Pred containerPred, code elementTypesCall,
                         string descr> :
    Type<
        And<[
            containerPred,
            Concat<
                "::llvm::all_of(" # elementTypesCall # ", [](::mlir::Type t) { "
                "return t && (",
                SubstLeaves<"$_self", "t", etype.predicate>,
                "); })"
            >
        ]>,
        descr # " with any combination of " # etype.summary # " values"> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypesCall = elementTypesCall;
}

// A Tuple that holds a mix of elements of the allowed types.
class TupleOf<list<Type> allowedTypes>
    : MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                         "::llvm::cast<::mlir::TupleType>($_self).getTypes()",
                         "tuple">;

// A Tuple with arbitrary nesting, where all elements are a mix of the allowed
// types.
class NestedTupleOf<list<Type> allowedTypes> :
    MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                       "getFlattenedTypes(::llvm::cast<::mlir::TupleType>($_self))",
                       "nested tuple">;

//===----------------------------------------------------------------------===//
// Common type constraints
//===----------------------------------------------------------------------===//
// Type constraint for types that are "like" some type or set of types T, that is
// they're either a T, a vector of Ts, or a tensor of Ts
class TypeOrContainer<Type allowedType, string name> : TypeConstraint<Or<[
  allowedType.predicate, VectorOf<[allowedType]>.predicate,
  TensorOf<[allowedType]>.predicate]>,
  name>;

// Temporary constraint to allow gradual transition to supporting 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
class TypeOrContainerOfAnyRank<Type allowedType, string name> : TypeConstraint<Or<[
  allowedType.predicate, VectorOfAnyRankOf<[allowedType]>.predicate,
  TensorOf<[allowedType]>.predicate]>,
  name>;


// Type constraint for bool-like types: bools, vectors of bools, tensors of
// bools.
def BoolLike : TypeOrContainer<I1, "bool-like">;

def BoolLikeOfAnyRank : TypeOrContainerOfAnyRank<I1, "bool-like">;

// Type constraint for signless-integer-like types: signless integers, indices,
// vectors of signless integers or indices, tensors of signless integers.
def SignlessIntegerLike : TypeOrContainer<AnySignlessIntegerOrIndex,
    "signless-integer-like">;

def SignlessIntegerLikeOfAnyRank : TypeOrContainerOfAnyRank<
    AnySignlessIntegerOrIndex,
    "signless-integer-like">;

// Type constraint for float-like types: floats, vectors or tensors thereof.
def FloatLike : TypeOrContainer<AnyFloat, "floating-point-like">;

// Type constraint for signless-integer-like or float-like types.
def SignlessIntegerOrFloatLike : TypeConstraint<Or<[
    SignlessIntegerLike.predicate, FloatLike.predicate]>,
    "signless-integer-like or floating-point-like">;

#endif // COMMON_TYPE_CONSTRAINTS_TD
