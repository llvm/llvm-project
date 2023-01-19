//===-- OpBase.td - Base op definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef OP_BASE
#define OP_BASE

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Common utilities for defining TableGen mechanisms
//===----------------------------------------------------------------------===//

// A workaround for the inability to define functions in Tablegen.
//
// The template parameter defines a string that can be extracted from an
// instance of this class by accessing the "result" member. Subclasses can take
// their own template parameters as function "arguments" and use them to
// populate result.
// For example, if it didn't already exist, a concat function could be defined
// like:
//
// class StrConcat<list<string> strings> :
//     StrFunc<!foldl("", strings, prev, cur, prev # cur)>
//
// and then called like
//
// StrConcat<["a", "b", "c"]>.result
//
// to get the string "abc"
class StrFunc<string r> {
  string result = r;
}

//===----------------------------------------------------------------------===//
// Predicate definitions
//===----------------------------------------------------------------------===//

// Base class for logical predicates.
//
// Predicates are used to compose constraints (see next section for details).
// There are two categories of predicates:
//
// 1. CPred: the primitive leaf predicate.
// 2. Compound predicate: a predicate composed from child predicates using
//    predicate combiners ("conjunction", "disjunction", "negation" or
//    "substitution").
class Pred;

// A logical predicate wrapping any C expression.
//
// This is the basis for composing more complex predicates. It is the "atom"
// predicate from the perspective of TableGen and the "interface" between
// TableGen and C++. What is inside is already C++ code, which will be treated
// as opaque strings with special placeholders to be substituted.
//
// ## Special placeholders
//
// Special placeholders can be used to refer to entities in the context where
// this predicate is used. They serve as "hooks" to the enclosing environment.
// The following special placeholders are supported in constraints for an op:
//
// * `$_builder` will be replaced by a mlir::Builder instance.
// * `$_op` will be replaced by the current operation.
// * `$_self` will be replaced with the entity this predicate is attached to.
//   E.g., `BoolAttr` is an attribute constraint that wraps a
//   `CPred<"$_self.isa<BoolAttr>()">` (see the following sections for details).
//   Then for `F32:$attr`,`$_self` will be replaced by `$attr`.
//   For type constraints, it's a little bit special since we want the
//   constraints on each type definition reads naturally and we want to attach
//   type constraints directly to an operand/result, $_self will be replaced
//   by the operand/result's type. E.g., for `F32` in `F32:$operand`, its
//   `$_self` will be expanded as `getOperand(...).getType()`.
//
// One thing to be noticed, while using these placeholders in the C expression,
// the type of placeholder is only guaranteed to be the base type. For example,
// if you have a predicate in the form `CPred<"CheckType($_self)">, the argument
// type of the function `CheckType` should be `mlir::Type`.
class CPred<code pred> : Pred {
  code predExpr = "(" # pred # ")";
}

// Kinds of predicate combiners.  These must closely match the predicates
// implemented by the C++ backend (tblgen::PredCombinerKind).
class PredCombinerKind;
def PredCombinerAnd : PredCombinerKind;
def PredCombinerOr : PredCombinerKind;
def PredCombinerNot : PredCombinerKind;
def PredCombinerSubstLeaves : PredCombinerKind;
def PredCombinerConcat : PredCombinerKind;

// A predicate that combines other predicates as defined by PredCombinerKind.
// Instantiated below.
class CombinedPred<PredCombinerKind k, list<Pred> c> : Pred {
  PredCombinerKind kind = k;
  list<Pred> children = c;
}

// Predicate combiners

// A predicate that holds if all of its children hold.  Always holds for zero
// children.
class And<list<Pred> children> : CombinedPred<PredCombinerAnd, children>;

// A predicate that holds if any of its children hold.  Never holds for zero
// children.
class Or<list<Pred> children> : CombinedPred<PredCombinerOr, children>;

// A predicate that holds if its child does not.
class Neg<Pred child> : CombinedPred<PredCombinerNot, [child]>;

// A predicate that substitutes "pat" with "repl" in predicate calls of the
// leaves of the predicate tree (i.e., not CombinedPred).
//
// This is plain string substitution without regular expressions or captures.
// New predicates with more complex logical can be introduced should the need
// arise.
class SubstLeaves<string pat, string repl, Pred child>
    : CombinedPred<PredCombinerSubstLeaves, [child]> {
  string pattern = pat;
  string replacement = repl;
}

// A predicate that prepends `pre` and appends `suf` to the final predicate
// string composed from `child`. This is plain string concatenation and there
// will be no substitution happening for `pre` and `suf`.
class Concat<string pre, Pred child, string suf> :
    CombinedPred<PredCombinerConcat, [child]> {
  string prefix = pre;
  string suffix = suf;
}

//===----------------------------------------------------------------------===//
// Constraint definitions
//===----------------------------------------------------------------------===//

// TODO: Merge Constraints into Pred.

// Base class for named constraints.
//
// An op's operands/attributes/results can have various requirements, e.g.,
// having certain types, having values inside a certain range, and so on.
// Besides, for a graph rewrite rule, the source pattern used to match against
// the existing graph has conditions, like the op's operand must be of a more
// constrained subtype, the attribute must have a certain value, and so on.
//
// These requirements and conditions are modeled using this class. Records of
// this class are used to generate verification code in op verifier, and
// matching code in pattern matcher.
//
// Constraints are predicates with descriptive names, to facilitate inspection,
// provide nice error messages, etc.
class Constraint<Pred pred, string desc = ""> {
  // The predicates that this constraint requires.
  Pred predicate = pred;
  // User-readable one line summary used in error reporting messages. If empty,
  // a generic message will be used.
  string summary = desc;
}

// Subclasses used to differentiate different constraint kinds. These are used
// as markers for the TableGen backend to handle different constraint kinds
// differently if needed. Constraints not deriving from the following subclasses
// are considered as uncategorized constraints.

// Subclass for constraints on a type.
class TypeConstraint<Pred predicate, string summary = "",
                     string cppClassNameParam = "::mlir::Type"> :
    Constraint<predicate, summary> {
  // The name of the C++ Type class if known, or Type if not.
  string cppClassName = cppClassNameParam;
}

// Subclass for constraints on an attribute.
class AttrConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// Subclass for constraints on a region.
class RegionConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// Subclass for constraints on a successor.
class SuccessorConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// How to use these constraint categories:
//
// * Use TypeConstraint to specify
//   * Constraints on an op's operand/result definition
//   * Further constraints to match an op's operand/result in source pattern
//
// * Use Attr (a subclass for AttrConstraint) for
//   * Constraints on an op's attribute definition
// * Use AttrConstraint to specify
//   * Further constraints to match an op's attribute in source pattern
//
// * Use uncategorized constraint to specify
//   * Multi-entity constraints in rewrite rules

//===----------------------------------------------------------------------===//
// Common predicates
//===----------------------------------------------------------------------===//

// Whether a type is a VectorType.
// Explicitly disallow 0-D vectors for now until we have good enough coverage.
def IsVectorTypePred : And<[CPred<"$_self.isa<::mlir::VectorType>()">,
                            CPred<"$_self.cast<::mlir::VectorType>().getRank() > 0">]>;

// Temporary vector type clone that allows gradual transition to 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
def IsVectorOfAnyRankTypePred : CPred<"$_self.isa<::mlir::VectorType>()">;

// Whether a type is a fixed-length VectorType.
def IsFixedVectorTypePred : CPred<[{$_self.isa<::mlir::VectorType>() &&
                                  !$_self.cast<VectorType>().isScalable()}]>;

// Whether a type is a scalable VectorType.
def IsScalableVectorTypePred : CPred<[{$_self.isa<::mlir::VectorType>() &&
                                   $_self.cast<VectorType>().isScalable()}]>;

// Whether a type is a TensorType.
def IsTensorTypePred : CPred<"$_self.isa<::mlir::TensorType>()">;

// Whether a type is a MemRefType.
def IsMemRefTypePred : CPred<"$_self.isa<::mlir::MemRefType>()">;

// Whether a type is an UnrankedMemRefType
def IsUnrankedMemRefTypePred
        : CPred<"$_self.isa<::mlir::UnrankedMemRefType>()">;

// Whether a type is an UnrankedTensorType
def IsUnrankedTensorTypePred
        : CPred<"$_self.isa<::mlir::UnrankedTensorType>()">;

// Whether a type is a BaseMemRefType
def IsBaseMemRefTypePred
        : CPred<"$_self.isa<::mlir::BaseMemRefType>()">;

// Whether a type is a ShapedType.
def IsShapedTypePred : CPred<"$_self.isa<::mlir::ShapedType>()">;

// For a ShapedType, verify that it has a static shape.
def HasStaticShapePred :
        CPred<"$_self.cast<::mlir::ShapedType>().hasStaticShape()">;

// Whether a type is a TupleType.
def IsTupleTypePred : CPred<"$_self.isa<::mlir::TupleType>()">;

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
class Variadic<Type type> : TypeConstraint<type.predicate, type.summary,
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
def NoneType : Type<CPred<"$_self.isa<::mlir::NoneType>()">, "none type",
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
def AnyInteger : Type<CPred<"$_self.isa<::mlir::IntegerType>()">, "integer",
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
def Index : Type<CPred<"$_self.isa<::mlir::IndexType>()">, "index",
                 "::mlir::IndexType">,
            BuildableType<"$_builder.getIndexType()">;

// Any signless integer type or index type.
def AnySignlessIntegerOrIndex : Type<CPred<"$_self.isSignlessIntOrIndex()">,
                                     "signless integer or index">;

// Floating point types.

// Any float type irrespective of its width.
def AnyFloat : Type<CPred<"$_self.isa<::mlir::FloatType>()">, "floating-point",
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
def F8E4M3FN : Type<CPred<"$_self.isFloat8E4M3FN()">, "f8E4M3FN type">,
               BuildableType<"$_builder.getFloat8E4M3FNType()">;
def F8E5M2 : Type<CPred<"$_self.isFloat8E5M2()">, "f8E5M2 type">,
             BuildableType<"$_builder.getFloat8E5M2Type()">;

def AnyComplex : Type<CPred<"$_self.isa<::mlir::ComplexType>()">,
                      "complex-type", "::mlir::ComplexType">;

class Complex<Type type>
    : ConfinedType<AnyComplex, [
          SubstLeaves<"$_self",
                      "$_self.cast<::mlir::ComplexType>().getElementType()",
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
def FunctionType : Type<CPred<"$_self.isa<::mlir::FunctionType>()">,
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
                "; }($_self.cast<::mlir::ShapedType>().getElementType())">]>,
         descr # " of " # AnyTypeOf<allowedTypes>.summary # " values", cppClassName>;

// Whether a shaped type is ranked.
def HasRankPred : CPred<"$_self.cast<::mlir::ShapedType>().hasRank()">;

// Whether a shaped type has one of the specified ranks.
class HasAnyRankOfPred<list<int> ranks> : And<[
    HasRankPred,
    Or<!foreach(rank, ranks,
                CPred<[{$_self.cast<::mlir::ShapedType>().getRank()
                         == }]
                      # rank>)>]>;

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
  ShapedContainerType<allowedTypes, IsScalableVectorTypePred,
          "scalable vector", "::mlir::VectorType">;

// Whether the number of elements of a vector is from the given
// `allowedRanks` list
class IsVectorOfRankPred<list<int> allowedRanks> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{$_self.cast<::mlir::VectorType>().getRank()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a fixed-length vector is from the given
// `allowedRanks` list
class IsFixedVectorOfRankPred<list<int> allowedRanks> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{$_self.cast<::mlir::VectorType>().getRank()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a scalable vector is from the given
// `allowedRanks` list
class IsScalableVectorOfRankPred<list<int> allowedRanks> :
  And<[IsScalableVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{$_self.cast<::mlir::VectorType>().getRank()
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
                   CPred<[{$_self.cast<::mlir::VectorType>().getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a fixed-length vector is from the given
// `allowedLengths` list
class IsFixedVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{$_self.cast<::mlir::VectorType>().getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a scalable vector is from the given
// `allowedLengths` list
class IsScalableVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsScalableVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{$_self.cast<::mlir::VectorType>().getNumElements()
                           == }]
                         # allowedlength>)>]>;

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

// Unranked tensor type whose element type is from the given
// `allowedTypes` list.
class UnrankedTensorOf<list<Type> allowedTypes>
  : ShapedContainerType<allowedTypes, IsUnrankedTensorTypePred,
      "unranked.tensor", "::mlir::UnrankedTensorType">;

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

class RankedTensorOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "ranked tensor">
  : TensorOf<allowedTypes, !listconcat([HasRankPred], preds), summary>;

def AnyRankedTensor : RankedTensorOf<[AnyType]>;

// Ranked tensor type with one of the specified types and ranks.
class TensorRankOf<list<Type> allowedTypes, list<int> ranks>
  : TensorOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tensor">;

class 0DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [0]>;
class 1DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [1]>;
class 2DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [2]>;
class 3DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [3]>;
class 4DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [4]>;

class StaticShapeTensorOf<list<Type> allowedTypes>
  : TensorOf<allowedTypes, [HasStaticShapePred], "statically shaped tensor">;

def AnyStaticShapeTensor : StaticShapeTensorOf<[AnyType]>;

//===----------------------------------------------------------------------===//
// Memref type.

// Unranked Memref type
class UnrankedMemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes,
                        IsUnrankedMemRefTypePred, "unranked.memref",
                        "::mlir::UnrankedMemRefType">;

def AnyUnrankedMemRef : UnrankedMemRefOf<[AnyType]>;

// Memrefs are blocks of data with fixed type and rank.
class MemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsMemRefTypePred, "memref",
                        "::mlir::MemRefType">;

def AnyMemRef : MemRefOf<[AnyType]>;

class RankedOrUnrankedMemRefOf<list<Type> allowedTypes>:
    AnyTypeOf<[UnrankedMemRefOf<allowedTypes>, MemRefOf<allowedTypes>]>;

def AnyRankedOrUnrankedMemRef: AnyTypeOf<[AnyUnrankedMemRef, AnyMemRef]>;

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
def HasStridesPred : CPred<[{ isStrided($_self.cast<::mlir::MemRefType>()) }]>;

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
                "::llvm::all_of(" # elementTypesCall # ", [](Type t) { "
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
                         "$_self.cast<::mlir::TupleType>().getTypes()",
                         "tuple">;

// A Tuple with arbitrary nesting, where all elements are a mix of the allowed
// types.
class NestedTupleOf<list<Type> allowedTypes> :
    MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                       "getFlattenedTypes($_self.cast<::mlir::TupleType>())",
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

//===----------------------------------------------------------------------===//
// Attribute definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base attribute definition

// Base class for all attributes.
class Attr<Pred condition, string summary = ""> :
    AttrConstraint<condition, summary> {
  code storageType = ?; // The backing mlir::Attribute type
  code returnType = ?;  // The underlying C++ value type

  // The call expression to convert from the storage type to the return
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format: $_self will be expanded to the attribute.
  //
  // For example, `$_self.getValue().getSExtValue()` for `IntegerAttr val` will
  // expand to `getAttrOfType<IntegerAttr>("val").getValue().getSExtValue()`.
  code convertFromStorage = "$_self.getValue()";

  // The call expression to build an attribute from a constant value.
  //
  // Format: $0 will be expanded to the constant value of the attribute.
  //
  // For example, `$_builder.getStringAttr("$0")` for `StringAttr:"foo"` will
  // expand to `builder.getStringAttr("foo")`.
  string constBuilderCall = ?;

  // Default value for attribute.
  // Requires a constBuilderCall defined.
  string defaultValue = ?;

  // The value type of this attribute. This corresponds to the mlir::Type that
  // this attribute returns via `getType()`.
  Type valueType = ?;

  // Whether the attribute is optional. Typically requires a custom
  // convertFromStorage method to handle the case where the attribute is
  // not present.
  bit isOptional = 0;

  // What is the base-level Attr instantiation that this Attr is built upon.
  // Unset means this is a base-level Attr.
  //
  // This field is used by attribute wrapper classes (DefaultValuedAttr,
  // OptionalAttr, etc.) to retrieve the base-level attribute definition.
  // This can be used for getting its name; otherwise, we will see
  // "anonymous_<number>" as the attribute def name because of template
  // instantiation.
  // TOOD(b/132458159): deduplicate the fields in attribute wrapper classes.
  Attr baseAttr = ?;

  // The fully-qualified C++ namespace where the generated class lives.
  string cppNamespace = "";

  // The full description of this attribute.
  string description = "";
}

// An attribute of a specific dialect.
class DialectAttr<Dialect d, Pred condition, string summary = ""> :
    Attr<condition, summary> {
  Dialect dialect = d;
  let cppNamespace = d.cppNamespace;
}

//===----------------------------------------------------------------------===//
// Attribute modifier definition

// Decorates an attribute to have an (unvalidated) default value if not present.
class DefaultValuedAttr<Attr attr, string val> :
    Attr<attr.predicate, attr.summary> {
  // Construct this attribute with the input attribute and change only
  // the default value.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = val;
  let valueType = attr.valueType;

  let baseAttr = attr;
}

// Decorates an optional attribute to have an (unvalidated) default value
// return by ODS generated accessors if not present.
class DefaultValuedOptionalAttr<Attr attr, string val> :
    Attr<attr.predicate, attr.summary> {
  // Construct this attribute with the input attribute and change only
  // the default value.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = val;
  let valueType = attr.valueType;
  let isOptional = 1;

  let baseAttr = attr;
}

// Decorates an attribute as optional. The return type of the generated
// attribute accessor method will be Optional<>.
class OptionalAttr<Attr attr> : Attr<attr.predicate, attr.summary> {
  // Rewrite the attribute to be optional.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = "::std::optional<" # attr.returnType #">";
  let convertFromStorage = "$_self ? " # returnType # "(" #
                           attr.convertFromStorage # ") : (::std::nullopt)";
  let valueType = attr.valueType;
  let isOptional = 1;

  let baseAttr = attr;
}

// Default-valued string-based attribute. Wraps the default value in escaped
// quotes.
class DefaultValuedStrAttr<Attr attr, string val>
    : DefaultValuedAttr<attr, "\"" # val # "\"">;
class DefaultValuedOptionalStrAttr<Attr attr, string val>
    : DefaultValuedOptionalAttr<attr, "\"" # val # "\"">;

//===----------------------------------------------------------------------===//
// Primitive attribute kinds

// A generic attribute that must be constructed around a specific buildable type
// `attrValType`. Backed by MLIR attribute kind `attrKind`.
class TypedAttrBase<Type attrValType, string attrKind, Pred condition,
                    string descr> :
    Attr<condition, descr> {
  let constBuilderCall = "$_builder.get" # attrKind # "(" #
                         attrValType.builderCall # ", $0)";
  let storageType = "::mlir::" # attrKind;
  let valueType = attrValType;
}

// Any attribute.
def AnyAttr : Attr<CPred<"true">, "any attribute"> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

// Any attribute from the given list
class AnyAttrOf<list<Attr> allowedAttrs, string summary = "",
                string cppClassName = "::mlir::Attribute",
                string fromStorage = "$_self"> : Attr<
    // Satisfy any of the allowed attribute's condition
    Or<!foreach(allowedattr, allowedAttrs, allowedattr.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedAttrs, t.summary), " or "),
        summary)> {
    let returnType = cppClassName;
    let convertFromStorage = fromStorage;
}

def BoolAttr : Attr<CPred<"$_self.isa<::mlir::BoolAttr>()">, "bool attribute"> {
  let storageType = [{ ::mlir::BoolAttr }];
  let returnType = [{ bool }];
  let valueType = I1;
  let constBuilderCall = "$_builder.getBoolAttr($0)";
}

// Index attribute.
def IndexAttr :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[CPred<"$_self.isa<::mlir::IntegerAttr>()">,
           CPred<"$_self.cast<::mlir::IntegerAttr>().getType()"
                 ".isa<::mlir::IndexType>()">]>,
      "index attribute"> {
  let returnType = [{ ::llvm::APInt }];
}

// Base class for any integer (regardless of signedness semantics) attributes
// of fixed width.
class AnyIntegerAttrBase<AnyI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"$_self.isa<::mlir::IntegerAttr>()">,
           CPred<"$_self.cast<::mlir::IntegerAttr>().getType()."
                 "isInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
  let constBuilderCall = ?;
}

def AnyI1Attr  : AnyIntegerAttrBase<AnyI1,  "1-bit integer attribute">;
def AnyI8Attr  : AnyIntegerAttrBase<AnyI8,  "8-bit integer attribute">;
def AnyI16Attr : AnyIntegerAttrBase<AnyI16, "16-bit integer attribute">;
def AnyI32Attr : AnyIntegerAttrBase<AnyI32, "32-bit integer attribute">;
def AnyI64Attr : AnyIntegerAttrBase<AnyI64, "64-bit integer attribute">;

def APIntAttr : Attr<CPred<"$_self.isa<::mlir::IntegerAttr>()">,
                     "arbitrary integer attribute"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::mlir::APInt }];
}

// Base class for signless integer attributes of fixed width.
class SignlessIntegerAttrBase<I attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"$_self.isa<::mlir::IntegerAttr>()">,
           CPred<"$_self.cast<::mlir::IntegerAttr>().getType()."
                 "isSignlessInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for signless integer attributes of fixed width that have a
// corresponding C++ type.
class TypedSignlessIntegerAttrBase<I attrValType, string retType, string descr>
    : SignlessIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getZExtValue()";
}

def I1Attr  : TypedSignlessIntegerAttrBase<
    I1,  "bool",     "1-bit signless integer attribute">;
def I8Attr  : TypedSignlessIntegerAttrBase<
    I8,  "uint8_t",  "8-bit signless integer attribute">;
def I16Attr : TypedSignlessIntegerAttrBase<
    I16, "uint16_t", "16-bit signless integer attribute">;
def I32Attr : TypedSignlessIntegerAttrBase<
    I32, "uint32_t", "32-bit signless integer attribute">;
def I64Attr : TypedSignlessIntegerAttrBase<
    I64, "uint64_t", "64-bit signless integer attribute">;

// Base class for signed integer attributes of fixed width.
class SignedIntegerAttrBase<SI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"$_self.isa<::mlir::IntegerAttr>()">,
           CPred<"$_self.cast<::mlir::IntegerAttr>().getType()."
                 "isSignedInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for signed integer attributes of fixed width that have a
// corresponding C++ type.
class TypedSignedIntegerAttrBase<SI attrValType, string retType, string descr>
    : SignedIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getSExtValue()";
}

def SI1Attr  : TypedSignedIntegerAttrBase<
    SI1,  "bool",    "1-bit signed integer attribute">;
def SI8Attr  : TypedSignedIntegerAttrBase<
    SI8,  "int8_t",  "8-bit signed integer attribute">;
def SI16Attr : TypedSignedIntegerAttrBase<
    SI16, "int16_t", "16-bit signed integer attribute">;
def SI32Attr : TypedSignedIntegerAttrBase<
    SI32, "int32_t", "32-bit signed integer attribute">;
def SI64Attr : TypedSignedIntegerAttrBase<
    SI64, "int64_t", "64-bit signed integer attribute">;

// Base class for unsigned integer attributes of fixed width.
class UnsignedIntegerAttrBase<UI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"$_self.isa<::mlir::IntegerAttr>()">,
           CPred<"$_self.cast<::mlir::IntegerAttr>().getType()."
                 "isUnsignedInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for unsigned integer attributes of fixed width that have a
// corresponding C++ type.
class TypedUnsignedIntegerAttrBase<UI attrValType, string retType, string descr>
    : UnsignedIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getZExtValue()";
}

def UI1Attr  : TypedUnsignedIntegerAttrBase<
    UI1,  "bool",     "1-bit unsigned integer attribute">;
def UI8Attr  : TypedUnsignedIntegerAttrBase<
    UI8,  "uint8_t",  "8-bit unsigned integer attribute">;
def UI16Attr : TypedUnsignedIntegerAttrBase<
    UI16, "uint16_t", "16-bit unsigned integer attribute">;
def UI32Attr : TypedUnsignedIntegerAttrBase<
    UI32, "uint32_t", "32-bit unsigned integer attribute">;
def UI64Attr : TypedUnsignedIntegerAttrBase<
    UI64, "uint64_t", "64-bit unsigned integer attribute">;

// Base class for float attributes of fixed width.
class FloatAttrBase<F attrValType, string descr> :
    TypedAttrBase<attrValType, "FloatAttr",
              And<[CPred<"$_self.isa<::mlir::FloatAttr>()">,
                     CPred<"$_self.cast<::mlir::FloatAttr>().getType().isF" #
                           attrValType.bitwidth # "()">]>,
              descr> {
  let returnType = [{ ::llvm::APFloat }];
}

def F32Attr : FloatAttrBase<F32, "32-bit float attribute">;
def F64Attr : FloatAttrBase<F64, "64-bit float attribute">;

// An attribute backed by a string type.
class StringBasedAttr<Pred condition, string descr> : Attr<condition, descr> {
  let constBuilderCall = "$_builder.getStringAttr($0)";
  let storageType = [{ ::mlir::StringAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
}

def StrAttr : StringBasedAttr<CPred<"$_self.isa<::mlir::StringAttr>()">,
                              "string attribute">;

// A string attribute that represents the name of a symbol.
def SymbolNameAttr : StringBasedAttr<CPred<"$_self.isa<::mlir::StringAttr>()">,
                                     "string attribute">;

// String attribute that has a specific value type.
class TypedStrAttr<Type ty>
    : StringBasedAttr<CPred<"$_self.isa<::mlir::StringAttr>()">,
                            "string attribute"> {
  let valueType = ty;
}

// Base class for attributes containing types. Example:
//   def IntTypeAttr : TypeAttrBase<"IntegerType", "integer type attribute">
// defines a type attribute containing an integer type.
class TypeAttrBase<string retType, string summary> :
    Attr<And<[
      CPred<"$_self.isa<::mlir::TypeAttr>()">,
      CPred<"$_self.cast<::mlir::TypeAttr>().getValue().isa<"
            # retType # ">()">]>,
    summary> {
  let storageType = [{ ::mlir::TypeAttr }];
  let returnType = retType;
  let valueType = NoneType;
  let convertFromStorage = "$_self.getValue().cast<" # retType # ">()";
}

def TypeAttr : TypeAttrBase<"::mlir::Type", "any type attribute"> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

class TypeAttrOf<Type ty>
   : TypeAttrBase<ty.cppClassName, "type attribute of " # ty.summary> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

// The mere presence of unit attributes has a meaning.  Therefore, unit
// attributes are always treated as optional and accessors to them return
// "true" if the attribute is present and "false" otherwise.
def UnitAttr : Attr<CPred<"$_self.isa<::mlir::UnitAttr>()">, "unit attribute"> {
  let storageType = [{ ::mlir::UnitAttr }];
  let constBuilderCall = "(($0) ? $_builder.getUnitAttr() : nullptr)";
  let convertFromStorage = "$_self != nullptr";
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// Composite attribute kinds

class DictionaryAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::DictionaryAttr }];
  let constBuilderCall = "$_builder.getDictionaryAttr($0)";
  let returnType = [{ ::mlir::DictionaryAttr }];
  let valueType = NoneType;
  let convertFromStorage = "$_self";
}

def DictionaryAttr
    : DictionaryAttrBase<CPred<"$_self.isa<::mlir::DictionaryAttr>()">,
                               "dictionary of named attribute values">;

class ElementsAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::ElementsAttr }];
  let returnType = [{ ::mlir::ElementsAttr }];
  let convertFromStorage = "$_self";
}

def ElementsAttr : ElementsAttrBase<CPred<"$_self.isa<::mlir::ElementsAttr>()">,
                                    "constant vector/tensor attribute">;

class IntElementsAttrBase<Pred condition, string summary> :
    ElementsAttrBase<And<[CPred<"$_self.isa<::mlir::DenseIntElementsAttr>()">,
                          condition]>,
                     summary> {
  let storageType = [{ ::mlir::DenseIntElementsAttr }];
  let returnType = [{ ::mlir::DenseIntElementsAttr }];

  let convertFromStorage = "$_self";
}

class DenseArrayAttrBase<string denseAttrName, string cppType, string summaryName> :
    ElementsAttrBase<CPred<"$_self.isa<::mlir::" # denseAttrName # ">()">,
                     summaryName # " dense array attribute"> {
  let storageType = "::mlir::" # denseAttrName;
  let returnType = "::llvm::ArrayRef<" # cppType # ">";
  let constBuilderCall = "$_builder.get" # denseAttrName # "($0)";
}
def DenseBoolArrayAttr : DenseArrayAttrBase<"DenseBoolArrayAttr", "bool", "i1">;
def DenseI8ArrayAttr : DenseArrayAttrBase<"DenseI8ArrayAttr", "int8_t", "i8">;
def DenseI16ArrayAttr : DenseArrayAttrBase<"DenseI16ArrayAttr", "int16_t", "i16">;
def DenseI32ArrayAttr : DenseArrayAttrBase<"DenseI32ArrayAttr", "int32_t", "i32">;
def DenseI64ArrayAttr : DenseArrayAttrBase<"DenseI64ArrayAttr", "int64_t", "i64">;
def DenseF32ArrayAttr : DenseArrayAttrBase<"DenseF32ArrayAttr", "float", "f32">;
def DenseF64ArrayAttr : DenseArrayAttrBase<"DenseF64ArrayAttr", "double", "f64">;

def IndexElementsAttr
    : IntElementsAttrBase<CPred<[{$_self.cast<::mlir::DenseIntElementsAttr>()
                                      .getType()
                                      .getElementType()
                                      .isIndex()}]>,
                          "index elements attribute">;

def AnyIntElementsAttr : IntElementsAttrBase<CPred<"true">, "integer elements attribute">;

class IntElementsAttrOf<int width> : IntElementsAttrBase<
  CPred<"$_self.cast<::mlir::DenseIntElementsAttr>().getType()."
        "getElementType().isInteger(" # width # ")">,
  width # "-bit integer elements attribute">;

def AnyI32ElementsAttr : IntElementsAttrOf<32>;
def AnyI64ElementsAttr : IntElementsAttrOf<64>;

class SignlessIntElementsAttr<int width> : IntElementsAttrBase<
  CPred<"$_self.cast<::mlir::DenseIntElementsAttr>().getType()."
        "getElementType().isSignlessInteger(" # width # ")">,
  width # "-bit signless integer elements attribute"> {

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({}, "
                                  "$_builder.getIntegerType(" # width # ")), "
    "::llvm::ArrayRef($0)).cast<::mlir::DenseIntElementsAttr>()";
}

def I32ElementsAttr : SignlessIntElementsAttr<32>;
def I64ElementsAttr : SignlessIntElementsAttr<64>;

// A `width`-bit signless integer elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class RankedSignlessIntElementsAttr<int width, list<int> dims> :
    SignlessIntElementsAttr<width> {
  // Check that this has the specified shape.
  let predicate = And<[
    SignlessIntElementsAttr<width>.predicate,
    CPred<"$_self.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == "
        "::mlir::ArrayRef<int64_t>({" # !interleave(dims, ", ") # "})">]>;

  let summary = width # "-bit signless int elements attribute of shape [" #
                !interleave(dims, ", ") # "]";

  let constBuilderCall = "::mlir::DenseIntElementsAttr::get("
    "::mlir::RankedTensorType::get({" # !interleave(dims, ", ") #
    "}, $_builder.getIntegerType(" # width # ")), ::llvm::ArrayRef($0))";
}

class RankedI32ElementsAttr<list<int> dims> :
    RankedSignlessIntElementsAttr<32, dims>;
class RankedI64ElementsAttr<list<int> dims> :
    RankedSignlessIntElementsAttr<64, dims>;

class FloatElementsAttr<int width> : ElementsAttrBase<
  CPred<"$_self.isa<::mlir::DenseFPElementsAttr>() &&"
      "$_self.cast<::mlir::DenseElementsAttr>().getType()."
      "getElementType().isF" # width # "()">,
  width # "-bit float elements attribute"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({}, $_builder.getF" # width # "Type()),"
    "::llvm::ArrayRef($0))";
  let convertFromStorage = "$_self";
}

def F64ElementsAttr : FloatElementsAttr<64>;

// A `width`-bit floating point elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class RankedFloatElementsAttr<int width, list<int> dims> : ElementsAttrBase<
  CPred<"$_self.isa<::mlir::DenseFPElementsAttr>() &&"
      "$_self.cast<::mlir::DenseFPElementsAttr>().getType()."
      "getElementType().isF" # width # "() && "
      // Check that this is ranked and has the specified shape.
      "$_self.cast<::mlir::DenseFPElementsAttr>().getType().hasRank() && "
      "$_self.cast<::mlir::DenseFPElementsAttr>().getType().getShape() == "
      "::mlir::ArrayRef<int64_t>({" # !interleave(dims, ", ") # "})">,
  width # "-bit float elements attribute of shape [" #
  !interleave(dims, ", ") # "]"> {

  let storageType = [{ ::mlir::DenseFPElementsAttr }];
  let returnType = [{ ::mlir::DenseFPElementsAttr }];

  let constBuilderCall = "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({" # !interleave(dims, ", ") #
    "}, $_builder.getF" # width # "Type()), "
    "::llvm::ArrayRef($0)).cast<::mlir::DenseFPElementsAttr>()";
  let convertFromStorage = "$_self";
}

class RankedF32ElementsAttr<list<int> dims> : RankedFloatElementsAttr<32, dims>;
class RankedF64ElementsAttr<list<int> dims> : RankedFloatElementsAttr<64, dims>;

def StringElementsAttr : ElementsAttrBase<
  CPred<"$_self.isa<::mlir::DenseStringElementsAttr>()" >,
  "string elements attribute"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

// Attributes containing affine maps.
def AffineMapAttr : Attr<
CPred<"$_self.isa<::mlir::AffineMapAttr>()">, "AffineMap attribute"> {
  let storageType = [{::mlir::AffineMapAttr }];
  let returnType = [{ ::mlir::AffineMap }];
  let valueType = Index;
  let constBuilderCall = "::mlir::AffineMapAttr::get($0)";
}

// Base class for array attributes.
class ArrayAttrBase<Pred condition, string summary> : Attr<condition, summary> {
  let storageType = [{ ::mlir::ArrayAttr }];
  let returnType = [{ ::mlir::ArrayAttr }];
  let valueType = NoneType;
  let convertFromStorage = "$_self";
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def ArrayAttr : ArrayAttrBase<CPred<"$_self.isa<::mlir::ArrayAttr>()">,
                              "array attribute">;

// Base class for array attributes whose elements are of the same kind.
// `element` specifies the element attribute kind stored in this array.
class TypedArrayAttrBase<Attr element, string summary>: ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"$_self.isa<::mlir::ArrayAttr>()">,
      // Guarantee all elements satisfy the constraints from `element`
      Concat<"::llvm::all_of($_self.cast<::mlir::ArrayAttr>(), "
                            "[&](::mlir::Attribute attr) { return attr && (",
                               SubstLeaves<"$_self", "attr", element.predicate>,
                            "); })">]>,
    summary> {

  Attr elementAttr = element;
}

def AffineMapArrayAttr : TypedArrayAttrBase<AffineMapAttr,
                                      "AffineMap array attribute"> {
  let constBuilderCall = "$_builder.getAffineMapArrayAttr($0)";
}

def BoolArrayAttr : TypedArrayAttrBase<BoolAttr,
                                      "1-bit boolean array attribute"> {
  let constBuilderCall = "$_builder.getBoolArrayAttr($0)";
}
def I32ArrayAttr : TypedArrayAttrBase<I32Attr,
                                      "32-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI32ArrayAttr($0)";
}
def I64ArrayAttr : TypedArrayAttrBase<I64Attr,
                                      "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}
// Variant of I64ArrayAttr whose user accessor is SmallVector<in64_t>.
def I64SmallVectorArrayAttr :
    TypedArrayAttrBase<I64Attr, "64-bit integer array attribute"> {
  let returnType = [{ ::llvm::SmallVector<int64_t, 8> }];
  let convertFromStorage = [{
    llvm::to_vector<4>(
      llvm::map_range($_self.getAsRange<mlir::IntegerAttr>(),
      [](IntegerAttr attr) { return attr.getInt(); }));
  }];
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}
def F32ArrayAttr : TypedArrayAttrBase<F32Attr, "32-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF32ArrayAttr($0)";
}
def F64ArrayAttr : TypedArrayAttrBase<F64Attr, "64-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF64ArrayAttr($0)";
}
def StrArrayAttr : TypedArrayAttrBase<StrAttr, "string array attribute"> {
  let constBuilderCall = "$_builder.getStrArrayAttr($0)";
}
def TypeArrayAttr : TypedArrayAttrBase<TypeAttr, "type array attribute"> {
  let constBuilderCall = "$_builder.getTypeArrayAttr($0)";
}
def IndexListArrayAttr :
  TypedArrayAttrBase<I64ArrayAttr, "Array of 64-bit integer array attributes">;
def DictArrayAttr :
  TypedArrayAttrBase<DictionaryAttr, "Array of dictionary attributes">;

// Attributes containing symbol references.
def SymbolRefAttr : Attr<CPred<"$_self.isa<::mlir::SymbolRefAttr>()">,
                        "symbol reference attribute"> {
  let storageType = [{ ::mlir::SymbolRefAttr }];
  let returnType = [{ ::mlir::SymbolRefAttr }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self";
}

def FlatSymbolRefAttr : Attr<CPred<"$_self.isa<::mlir::FlatSymbolRefAttr>()">,
                                   "flat symbol reference attribute"> {
  let storageType = [{ ::mlir::FlatSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def SymbolRefArrayAttr :
  TypedArrayAttrBase<SymbolRefAttr, "symbol ref array attribute"> {
  let constBuilderCall = ?;
}

def FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// Derive attribute kinds

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
// Note: All derived attributes should be materializable as an Attribute. E.g.,
// do not use DerivedAttr for things that could not have been stored as
// Attribute.
//
class DerivedAttr<code ret, code b, code convert = ""> :
    Attr<CPred<"true">, "derived attribute"> {
  let returnType = ret;
  code body = b;

  // Specify how to convert from the derived attribute to an attribute.
  //
  // ## Special placeholders
  //
  // Special placeholders can be used to refer to entities during conversion:
  //
  // * `$_builder` will be replaced by a mlir::Builder instance.
  // * `$_ctxt` will be replaced by the MLIRContext* instance.
  // * `$_self` will be replaced with the derived attribute (value produces
  //    `returnType`).
  let convertFromStorage = convert;
}

// Derived attribute that returns a mlir::Type.
class DerivedTypeAttr<code body> : DerivedAttr<"::mlir::Type", body> {
  let convertFromStorage = "::mlir::TypeAttr::get($_self)";
}

//===----------------------------------------------------------------------===//
// Constant attribute kinds

// Represents a constant attribute of specific Attr type. A constant
// attribute can be specified only of attributes that have a constant
// builder call defined. The constant value is specified as a string.
//
// If used as a constraint, it generates a matcher on a constant attribute by
// using the constant value builder of the attribute and the value.
class ConstantAttr<Attr attribute, string val> : AttrConstraint<
    CPred<"$_self == " # !subst("$0", val, attribute.constBuilderCall)>,
    "constant attribute " # val> {
  Attr attr = attribute;
  string value = val;
}

class ConstF32Attr<string val> : ConstantAttr<F32Attr, val>;
def ConstBoolAttrFalse : ConstantAttr<BoolAttr, "false">;
def ConstBoolAttrTrue : ConstantAttr<BoolAttr, "true">;
def ConstUnitAttr : ConstantAttr<UnitAttr, "true">;

// Constant string-based attribute. Wraps the desired string in escaped quotes.
class ConstantStrAttr<Attr attribute, string val>
    : ConstantAttr<attribute, "\"" # val # "\"">;

//===----------------------------------------------------------------------===//
// Common attribute constraints
//===----------------------------------------------------------------------===//

// A general mechanism to further confine the given `attr` with all the
// `constraints`. This allows to compose complex constraints out of a series
// of more primitive ones.
class ConfinedAttr<Attr attr, list<AttrConstraint> constraints> : Attr<
    And<!listconcat([attr.predicate],
                      !foreach(pred, constraints, pred.predicate))>,
    !foldl(/*init*/attr.summary, /*list*/constraints,
           prev, cur, prev # " " # cur.summary)> {
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = attr.defaultValue;
  let valueType = attr.valueType;
  let isOptional = attr.isOptional;

  let baseAttr = attr;
}

// An AttrConstraint that holds if all attr constraints specified in
// 'constraints' hold.
class AllAttrOf<list<AttrConstraint> constraints> : AttrConstraint<
    And<!listconcat([!head(constraints).predicate],
                    !foreach(pred, !tail(constraints), pred.predicate))>,
    !interleave(!foreach(con, constraints, con.summary), " and ")> {
}

class IntMinValue<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::IntegerAttr>().getInt() >= " # n>,
    "whose minimum value is " # n>;

class IntMaxValue<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::IntegerAttr>().getInt() <= " # n>,
    "whose maximum value is " # n>;

def IntNonNegative : AttrConstraint<
    CPred<"!$_self.cast<::mlir::IntegerAttr>().getValue().isNegative()">,
    "whose value is non-negative">;

def IntPositive : AttrConstraint<
    CPred<"$_self.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()">,
    "whose value is positive">;

class ArrayMinCount<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::ArrayAttr>().size() >= " # n>,
    "with at least " # n # " elements">;

class ArrayCount<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::ArrayAttr>().size() == " #n>,
    "with exactly " # n # " elements">;

class DenseArrayCount<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::DenseArrayAttr>().size() == " #n>,
    "with exactly " # n # " elements">;

class DenseArrayStrictlyPositive<DenseArrayAttrBase arrayType> : AttrConstraint<
  CPred<"::llvm::all_of($_self.cast<" # arrayType #">().asArrayRef(), "
                        "[&](auto v) { return v > 0; })">,
  "whose value is positive">;

class DenseArrayNonNegative<DenseArrayAttrBase arrayType> : AttrConstraint<
  CPred<"::llvm::all_of($_self.cast<" # arrayType #">().asArrayRef(), "
                        "[&](auto v) { return v >= 0; })">,
  "whose value is non-negative">;

class DenseArraySorted<DenseArrayAttrBase arrayType> : AttrConstraint<
    CPred<"llvm::is_sorted($_self.cast<" # arrayType # ">().asArrayRef())">,
    "should be in non-decreasing order">;

class DenseArrayStrictlySorted<DenseArrayAttrBase arrayType> : AttrConstraint<
    And<[
      CPred<"llvm::is_sorted($_self.cast<" # arrayType # ">().asArrayRef())">,
      // Check that no two adjacent elements are the same.
      CPred<"[](" # arrayType.returnType # " a) {\n"
        "return std::adjacent_find(std::begin(a), std::end(a)) == "
        "std::end(a);\n"
        "}($_self.cast<" # arrayType # ">().asArrayRef())"
      >]>,
    "should be in increasing order">;

class IntArrayNthElemEq<int index, int value> : AttrConstraint<
    And<[
      CPred<"$_self.cast<::mlir::ArrayAttr>().size() > " # index>,
      CPred<"$_self.cast<::mlir::ArrayAttr>()[" # index # "]"
        ".cast<::mlir::IntegerAttr>().getInt() == " # value>
       ]>,
    "whose " # index # "-th element must be " # value>;

class IntArrayNthElemMinValue<int index, int min> : AttrConstraint<
    And<[
      CPred<"$_self.cast<::mlir::ArrayAttr>().size() > " # index>,
      CPred<"$_self.cast<::mlir::ArrayAttr>()[" # index # "]"
        ".cast<::mlir::IntegerAttr>().getInt() >= " # min>
        ]>,
    "whose " # index # "-th element must be at least " # min>;

def IsNullAttr : AttrConstraint<
    CPred<"!$_self">, "empty attribute (for optional attributes)">;

//===----------------------------------------------------------------------===//
// Region definitions
//===----------------------------------------------------------------------===//

class Region<Pred condition, string descr = ""> :
    RegionConstraint<condition, descr>;

// Any region.
def AnyRegion : Region<CPred<"true">, "any region">;

// A region with the given number of blocks.
class SizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItems($_self, " # numBlocks # ")">,
  "region with " # numBlocks # " blocks">;

// A region with at least the given number of blocks.
class MinSizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItemsOrMore($_self, " # numBlocks # ")">,
  "region with at least " # numBlocks # " blocks">;

// A variadic region constraint. It expands to zero or more of the base region.
class VariadicRegion<Region region>
  : Region<region.predicate, region.summary>;

//===----------------------------------------------------------------------===//
// Successor definitions
//===----------------------------------------------------------------------===//

class Successor<Pred condition, string descr = ""> :
    SuccessorConstraint<condition, descr>;

// Any successor.
def AnySuccessor : Successor<?, "any successor">;

// A variadic successor constraint. It expands to zero or more of the base
// successor.
class VariadicSuccessor<Successor successor>
  : Successor<successor.predicate, successor.summary>;


//===----------------------------------------------------------------------===//
// Trait definitions
//===----------------------------------------------------------------------===//

// Trait represents a trait regarding an attribute, operation, or type.
class Trait;

// Define a Trait corresponding to a list of Traits, this allows for specifying
// a list of traits as trait. Avoids needing to do `[Traits, ...] # ListOfTraits
// # [Others, ...]` while still allowing providing convenient groupings.
class TraitList<list<Trait> props> : Trait {
  list<Trait> traits = props;
}

// NativeTrait corresponds to the MLIR C++ trait mechanism. The purpose to wrap
// around C++ symbol string with this class is to make traits specified for
// entities in TableGen less alien and more integrated.
class NativeTrait<string name, string entityType> : Trait {
  string trait = name;
  string cppNamespace = "::mlir::" # entityType # "Trait";
}

// ParamNativeTrait corresponds to the template-parameterized traits in the C++
// implementation. MLIR uses nested class templates to implement such traits
// leading to constructs of the form "TraitName<Parameters>::Impl". Use the
// value in `prop` as the trait name and the value in `params` as parameters to
// construct the native trait class name.
class ParamNativeTrait<string prop, string params, string entityType>
    : NativeTrait<prop # "<" # params # ">::Impl", entityType>;

// GenInternalTrait is a trait that does not have direct C++ mapping but affects
// an entities definition generator internals, like how operation builders and
// operand/attribute/result getters are generated.
class GenInternalTrait<string prop, string entityType> : Trait {
  string trait = "::mlir::" # entityType # "Trait::" # prop;
}

// PredTrait is a trait implemented by way of a predicate on an entity.
class PredTrait<string descr, Pred pred> : Trait {
  string summary = descr;
  Pred predicate = pred;
}

//===----------------------------------------------------------------------===//
// OpTrait definitions
//===----------------------------------------------------------------------===//

// A trait that describes the structure of operation will be marked with
// `StructuralOpTrait` and they will be verified first.
class StructuralOpTrait;

// These classes are used to define operation specific traits.
class NativeOpTrait<string name, list<Trait> traits = []>
    : NativeTrait<name, "Op"> {
  // Specify the list of traits that need to be verified before the verification
  // of this NativeOpTrait.
  list<Trait> dependentTraits = traits;
}
class ParamNativeOpTrait<string prop, string params,
                         list<Trait> traits = []>
    : ParamNativeTrait<prop, params, "Op"> {
  // Specify the list of traits that need to be verified before the verification
  // of this ParamNativeOpTrait.
  list<Trait> dependentTraits = traits;
}
class GenInternalOpTrait<string prop, list<Trait> traits = []>
    : GenInternalTrait<prop, "Op"> {
  // Specify the list of traits that need to be verified before the verification
  // of this GenInternalOpTrait.
  list<Trait> dependentTraits = traits;
}
class PredOpTrait<string descr, Pred pred, list<Trait> traits = []>
    : PredTrait<descr, pred> {
  // Specify the list of traits that need to be verified before the verification
  // of this PredOpTrait.
  list<Trait> dependentTraits = traits;
}

// Op defines an affine scope.
def AffineScope : NativeOpTrait<"AffineScope">;
// Op defines an automatic allocation scope.
def AutomaticAllocationScope :
  NativeOpTrait<"AutomaticAllocationScope">;
// Op supports operand broadcast behavior.
def ResultsBroadcastableShape :
  NativeOpTrait<"ResultsBroadcastableShape">;
// X op Y == Y op X
def Commutative  : NativeOpTrait<"IsCommutative">;
// op op X == op X (unary) / X op X == X (binary)
// FIXME: Idempotent should depend on SameOperandsAndResultType
def Idempotent : NativeOpTrait<"IsIdempotent">;
// op op X == X
// FIXME: Involution should depend on SameOperandsAndResultType
def Involution : NativeOpTrait<"IsInvolution">;
// Op behaves like a constant.
def ConstantLike : NativeOpTrait<"ConstantLike">;
// Op is isolated from above.
def IsolatedFromAbove : NativeOpTrait<"IsIsolatedFromAbove">;
// Op results are float or vectors/tensors thereof.
def ResultsAreFloatLike : NativeOpTrait<"ResultsAreFloatLike">;
// Op has the same operand type.
def SameTypeOperands : NativeOpTrait<"SameTypeOperands">;
// Op has same shape for all operands.
def SameOperandsShape : NativeOpTrait<"SameOperandsShape">;
// Op has same operand and result shape.
def SameOperandsAndResultShape :
  NativeOpTrait<"SameOperandsAndResultShape">;
// Op has the same element type (or type itself, if scalar) for all operands.
def SameOperandsElementType :
  NativeOpTrait<"SameOperandsElementType">;
// Op has the same operand and result element type (or type itself, if scalar).
def SameOperandsAndResultElementType :
  NativeOpTrait<"SameOperandsAndResultElementType">;
// Op is a terminator.
def Terminator : NativeOpTrait<"IsTerminator">;
// Op can be safely normalized in the presence of MemRefs with
// non-identity maps.
def MemRefsNormalizable : NativeOpTrait<"MemRefsNormalizable">;
// Op is elementwise on tensor/vector operands and results.
def Elementwise : NativeOpTrait<"Elementwise">;
// Elementwise op can be applied to scalars instead tensor/vector operands.
def Scalarizable : NativeOpTrait<"Scalarizable", [Elementwise]>;
// Elementwise op can be applied to all-vector operands.
def Vectorizable : NativeOpTrait<"Vectorizable", [Elementwise]>;
// Elementwise op can be applied to all-tensor operands.
def Tensorizable : NativeOpTrait<"Tensorizable", [Elementwise]>;

// Group together `Elementwise`, `Scalarizable`, `Vectorizable`, and
// `Tensorizable` for convenience.
def ElementwiseMappable : TraitList<[
    Elementwise,
    Scalarizable,
    Vectorizable,
    Tensorizable,
]>;

// Op's regions have a single block.
def SingleBlock : NativeOpTrait<"SingleBlock">, StructuralOpTrait;

// Op's regions have a single block with the specified terminator.
class SingleBlockImplicitTerminator<string op>
    : ParamNativeOpTrait<"SingleBlockImplicitTerminator", op>,
      StructuralOpTrait;

// Op's regions don't have terminator.
def NoTerminator : NativeOpTrait<"NoTerminator">, StructuralOpTrait;

// Op's parent operation is the provided one.
class HasParent<string op>
    : ParamNativeOpTrait<"HasParent", op>, StructuralOpTrait;

class ParentOneOf<list<string> ops>
    : ParamNativeOpTrait<"HasParent", !interleave(ops, ", ")>,
      StructuralOpTrait;

// Op result type is derived from the first attribute. If the attribute is an
// subclass of `TypeAttrBase`, its value is used, otherwise, the type of the
// attribute content is used.
def FirstAttrDerivedResultType :
  GenInternalOpTrait<"FirstAttrDerivedResultType">;

// TODO: Turn the following into normal traits and generate verification for
// them.

// All variadic operands of the op have the same number of values.
// A variadic operand contains an array of values whose array size is only
// known at runtime. This trait requires all variadic operands of an op
// to have the same array size.
def SameVariadicOperandSize : GenInternalOpTrait<"SameVariadicOperandSize">;
// All variadic results of the op have the same number of values.
// A variadic result contains an array of values whose array size is only
// known at runtime. This trait requires all variadic results of an op
// to have the same array size.
def SameVariadicResultSize : GenInternalOpTrait<"SameVariadicResultSize">;

// Uses an attribute named `operand_segment_sizes` to specify how many actual
// operand each ODS-declared operand (variadic or not) corresponds to.
// This trait is used for ops that have multiple variadic operands but do
// not know statically their size relationship. The attribute must be a 1D
// vector that has the same number of elements as the number of ODS declared
// operands. That means even if some operands are non-variadic, the attribute
// still need to have an element for its size, which is always 1.
def AttrSizedOperandSegments :
  NativeOpTrait<"AttrSizedOperandSegments">, StructuralOpTrait;
// Similar to AttrSizedOperandSegments, but used for results. The attribute
// should be named as `result_segment_sizes`.
def AttrSizedResultSegments  :
  NativeOpTrait<"AttrSizedResultSegments">, StructuralOpTrait;

// Op attached regions have no arguments
def NoRegionArguments : NativeOpTrait<"NoRegionArguments">, StructuralOpTrait;

//===----------------------------------------------------------------------===//
// OpInterface definitions
//===----------------------------------------------------------------------===//

// Marker used to identify the argument list for an op or interface method.
def ins;

// This class represents a typed argument with optional default value for C
// function signatures, e.g. builders or methods.
class CArg<string ty, string value = ""> {
  string type = ty;
  string defaultValue = value;
}

// InterfaceTrait corresponds to a specific 'Interface' class defined in C++.
// The purpose to wrap around C++ symbol string with this class is to make
// interfaces specified for ops in TableGen less alien and more integrated.
class InterfaceTrait<string name> : NativeTrait<"", ""> {
  let trait = name # "::Trait";
  let cppNamespace = "";

  // An optional code block containing extra declarations to place in the
  // interface trait declaration.
  code extraTraitClassDeclaration = "";
}

// OpInterfaceTrait corresponds to a specific 'OpInterface' class defined in
// C++. The purpose to wrap around C++ symbol string with this class is to make
// interfaces specified for ops in TableGen less alien and more integrated.
class OpInterfaceTrait<string name, code verifyBody = [{}],
                       list<Trait> traits = []>
    : InterfaceTrait<name> {
  // Specify the body of the verification function. `$_op` will be replaced with
  // the operation being verified.
  code verify = verifyBody;

  // A bit indicating if the verifier needs to access the ops in the regions. If
  // it set to `1`, the region ops will be verified before invoking this
  // verifier.
  bit verifyWithRegions = 0;

  // Specify the list of traits that need to be verified before the verification
  // of this OpInterfaceTrait.
  list<Trait> dependentTraits = traits;
}

// This class represents a single, optionally static, interface method.
// Note: non-static interface methods have an implicit parameter, either
// $_op/$_attr/$_type corresponding to an instance of the derived value.
class InterfaceMethod<string desc, string retTy, string methodName,
                      dag args = (ins), code methodBody = [{}],
                      code defaultImplementation = [{}]> {
  // A human-readable description of what this method does.
  string description = desc;

  // The name of the interface method.
  string name = methodName;

  // The c++ type-name of the return type.
  string returnType = retTy;

  // A dag of string that correspond to the arguments of the method.
  dag arguments = args;

  // An optional body to the method.
  code body = methodBody;

  // An optional default implementation of the method.
  code defaultBody = defaultImplementation;
}

// This class represents a single static interface method.
class StaticInterfaceMethod<string desc, string retTy, string methodName,
                            dag args = (ins), code methodBody = [{}],
                            code defaultImplementation = [{}]>
    : InterfaceMethod<desc, retTy, methodName, args, methodBody,
                      defaultImplementation>;

// Interface represents a base interface.
class Interface<string name> {
  // A human-readable description of what this interface does.
  string description = "";

  // The name given to the c++ interface class.
  string cppInterfaceName = name;

  // The C++ namespace that this interface should be placed into.
  //
  // To specify nested namespaces, use "::" as the delimiter, e.g., given
  // "A::B", ops will be placed in `namespace A { namespace B { <def> } }`.
  string cppNamespace = "";

  // The list of methods defined by this interface.
  list<InterfaceMethod> methods = [];

  // An optional code block containing extra declarations to place in the
  // interface declaration.
  code extraClassDeclaration = "";

  // An optional code block containing extra declarations to place in both
  // the interface and trait declaration.
  code extraSharedClassDeclaration = "";

  // An optional code block for adding additional "classof" logic. This can
  // be used to better enable "optional" interfaces, where an entity only
  // implements the interface if some dynamic characteristic holds.
  // `$_attr`/`$_op`/`$_type` may be used to refer to an instance of the
  // entity being checked.
  code extraClassOf = "";
}

// AttrInterface represents an interface registered to an attribute.
class AttrInterface<string name> : Interface<name>, InterfaceTrait<name>,
	Attr<CPred<"$_self.isa<"
		# !if(!empty(cppNamespace),"", cppNamespace # "::") # name # ">()">,
			name # " instance">
{
	let storageType = !if(!empty(cppNamespace), "", cppNamespace # "::") # name;
	let returnType = storageType;
	let convertFromStorage = "$_self";
}

// OpInterface represents an interface registered to an operation.
class OpInterface<string name> : Interface<name>, OpInterfaceTrait<name>;

// TypeInterface represents an interface registered to a type.
class TypeInterface<string name> : Interface<name>, InterfaceTrait<name>,
	Type<CPred<"$_self.isa<"
		# !if(!empty(cppNamespace),"", cppNamespace # "::") # name # ">()">,
			name # " instance",
				!if(!empty(cppNamespace),"", cppNamespace # "::") # name>;

// Whether to declare the interface methods in the user entity's header. This
// class simply wraps an Interface but is used to indicate that the method
// declarations should be generated. This class takes an optional set of methods
// that should have declarations generated even if the method has a default
// implementation.
class DeclareInterfaceMethods<list<string> overridenMethods = []> {
    // This field contains a set of method names that should always have their
    // declarations generated. This allows for generating declarations for
    // methods with default implementations that need to be overridden.
    list<string> alwaysOverriddenMethods = overridenMethods;
}
class DeclareAttrInterfaceMethods<AttrInterface interface,
                                  list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        AttrInterface<interface.cppInterfaceName> {
    let description = interface.description;
    let cppInterfaceName = interface.cppInterfaceName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
}
class DeclareOpInterfaceMethods<OpInterface interface,
                                list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        OpInterface<interface.cppInterfaceName> {
    let description = interface.description;
    let cppInterfaceName = interface.cppInterfaceName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
}
class DeclareTypeInterfaceMethods<TypeInterface interface,
                                  list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        TypeInterface<interface.cppInterfaceName> {
    let description = interface.description;
    let cppInterfaceName = interface.cppInterfaceName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
}

//===----------------------------------------------------------------------===//
// Op definitions
//===----------------------------------------------------------------------===//

// Marker used to identify the result list for an op.
def outs;

// Marker used to identify the region list for an op.
def region;

// Marker used to identify the successor list for an op.
def successor;

// Class for defining a custom builder.
//
// TableGen generates several generic builders for each op by default (see
// comment in the `Op` class). If the default generated ones cannot cover
// some use case, custom builders can be defined using instances of this class.
//
// The signature of the builder is always
//
// ```c++
// static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
//                   <other-parameters>...) {
//   <body>...
// }
// ```
//
// To define a custom builder, the parameter list (*excluding* the
// `OpBuilder &builder, OperationState &state` part) and body should be passed
// in as separate template arguments to this class. The parameter list is a
// TableGen DAG with `ins` operation with named arguments, which has either:
//   - string initializers ("Type":$name) to represent a typed parameter, or
//   - CArg-typed initializers (CArg<"Type", "default">:$name) to represent a
//     typed parameter that may have a default value.
// The type string is used verbatim to produce code and, therefore, must be a
// valid C++ type. It is used inside the C++ namespace of the parent Op's
// dialect; explicit namespace qualification like `::mlir` may be necessary if
// Ops are not placed inside the `mlir` namespace. The default value string is
// used verbatim to produce code and must be a valid C++ initializer the given
// type. For example, the following signature specification
//
// ```
// OpBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg)>
// ```
//
// has an integer parameter and a float parameter with a default value.
//
// If an empty string is passed in for `body`, then *only* the builder
// declaration will be generated; this provides a way to define complicated
// builders entirely in C++.
class OpBuilder<dag p, code b = ""> {
  dag dagParams = p;
  code body = b;
}

// A base decorator class that may optionally be added to OpVariables.
class OpVariableDecorator;

// Class for providing additional information on the variables, i.e. arguments
// and results, of an operation.
class OpVariable<Constraint varConstraint, string desc = "",
                 list<OpVariableDecorator> varDecorators = []> {
  // The constraint, either attribute or type, of the argument.
  Constraint constraint = varConstraint;

  // One-line human-readable description of the argument.
  string summary = desc;

  // The list of decorators for this variable, e.g. side effects.
  list<OpVariableDecorator> decorators = varDecorators;
}
class Arg<Constraint constraint, string desc = "",
          list<OpVariableDecorator> decorators = []>
  : OpVariable<constraint, desc, decorators>;
class Res<Constraint constraint, string desc = "",
          list<OpVariableDecorator> decorators = []>
  : OpVariable<constraint, desc, decorators>;

// Base class for all ops.
class Op<Dialect dialect, string mnemonic, list<Trait> props = []> {
  // The dialect of the op.
  Dialect opDialect = dialect;

  // The mnemonic of the op.
  string opName = mnemonic;

  // The C++ namespace to use for this op.
  string cppNamespace = dialect.cppNamespace;

  // One-line human-readable description of what the op does.
  string summary = "";

  // Additional, longer human-readable description of what the op does.
  string description = "";

  // Dag containing the arguments of the op. Default to 0 arguments.
  dag arguments = (ins);

  // The list of results of the op. Default to 0 results.
  dag results = (outs);

  // The list of regions of the op. Default to 0 regions.
  dag regions = (region);

  // The list of successors of the op. Default to 0 successors.
  dag successors = (successor);

  // Attribute getters can be added to the op by adding an Attr member
  // with the name and type of the attribute. E.g., adding int attribute
  // with name "value" and type "i32":
  //   I32Attr value;

  // Define the hooks used for building, parsing, printing, verification.

  // Custom builder.
  // In addition to the custom builder provided here, and unless
  // skipDefaultBuilders is set, two default builders are generated, with the
  // following signatures:
  //
  // ```c++
  // static void build(OpBuilder &, OperationState &odsState,
  //                   Type <result0-name>, Type <result1-name>, ...,
  //                   Value <arg0-name>, Value <arg1-name>, ...,
  //                   Attribute <attr0-name>, Attribute <attr1-name>, ...);
  // ```
  // * where the attributes follow the same declaration order as in the op.
  //
  // ```c++
  // static void build(OpBuilder &, OperationState &odsState,
  //                   TypeRange resultTypes,
  //                   ValueRange operands,
  //                   ArrayRef<NamedAttribute> attributes);
  // ```
  list<OpBuilder> builders = ?;

  // Avoid generating default build functions.  Custom builders must be
  // provided.
  bit skipDefaultBuilders = 0;

  // Custom assembly format.
  /// This field corresponds to a declarative description of the assembly format
  /// for this operation. If populated, the `hasCustomAssemblyFormat` field is
  /// ignored.
  string assemblyFormat = ?;
  /// This field indicates that the operation has a custom assembly format
  /// implemented in C++. When set to `1` a `parse` and `print` method are generated
  /// on the operation class. The operation should implement these methods to
  /// support the custom format of the operation. The methods have the form:
  ///   * ParseResult parse(OpAsmParser &parser, OperationState &result)
  ///   * void print(OpAsmPrinter &p)
  bit hasCustomAssemblyFormat = 0;

  // A bit indicating if the operation has additional invariants that need to
  // verified (aside from those verified by other ODS constructs). If set to `1`,
  // an additional `LogicalResult verify()` declaration will be generated on the
  // operation class. The operation should implement this method and verify the
  // additional necessary invariants. This verifier shouldn't access any nested
  // operations because those operations may ill-formed. Use the
  // `hasRegionVerifier` below instead.
  bit hasVerifier = 0;

  // A bit indicating if the operation has additional invariants that need to
  // verified and which associate with regions (aside from those verified by the
  // traits). If set to `1`, an additional `LogicalResult verifyRegions()`
  // declaration will be generated on the operation class. The operation should
  // implement this method and verify the additional necessary invariants
  // associated with regions. Note that this method is invoked after all the
  // region ops are verified.
  bit hasRegionVerifier = 0;

  // Whether this op has associated canonicalization patterns.
  bit hasCanonicalizer = 0;

  // Whether this op has a static "canonicalize" method to perform "match and
  // rewrite patterns".
  bit hasCanonicalizeMethod = 0;

  // Whether this op has a folder.
  bit hasFolder = 0;

  // Op traits.
  // Note: The list of traits will be uniqued by ODS.
  list<Trait> traits = props;

  // Additional code that will be added to the public part of the generated
  // C++ code of the op declaration.
  code extraClassDeclaration = ?;

  // Additional code that will be added to the generated source file. The
  // generated code is placed inside the op's C++ namespace. `$cppClass` is
  // replaced by the op's C++ class name.
  code extraClassDefinition = ?;
}

// The arguments of an op.
class Arguments<dag args> {
  dag arguments = args;
}

// The results of an op.
class Results<dag rets> {
  dag results = rets;
}

//===----------------------------------------------------------------------===//
// Common op type constraints
//===----------------------------------------------------------------------===//

// These traits are for verifying properties of an op that require knowledge of
// multiple arguments or results. For verifying properties of a single argument
// or result, prefer operand type constraints.

// These traits often require including "mlir/IR/TypeUtilities.h".

// TODO: Improve the autogenerated error messages.

class Rank<string name> :
    StrFunc<"$" # name # ".getType().cast<::mlir::ShapedType>().getRank()">;

class Shape<string name> :
    StrFunc<"$" # name # ".getType().cast<::mlir::ShapedType>().getShape()">;

class ElementCount<string name> :
  StrFunc<"$" # name # ".getType().cast<::mlir::ShapedType>()"
                                 ".getNumElements()">;

class ElementType<string name> : StrFunc<"getElementTypeOrSelf($" # name # ")">;

class AllMatchPred<list<string> values> :
  CPred<!if(!lt(!size(values), 2),
            "true",
            !foldl("(" # !head(values) # ")", !tail(values), acc, v,
                   acc # " == (" # v # ") && (" # v # ")")
              # " == (" # !head(values) # ")")>;

class AllMatch<list<string> values, string summary> :
    PredOpTrait<summary, AllMatchPred<values>>;

// TODO: Only works for non-variadic.
class AllMatchSameOperatorPred<list<string> names, string operator> :
    AllMatchPred<!foreach(n, names, !subst("$_self", "$" # n, operator))>;

class AllMatchSameOperatorTrait<list<string> names, string operator,
                                string summary> :
    PredOpTrait<
        "all of {" # !interleave(names, ", ") # "} have same " # summary,
        AllMatchSameOperatorPred<names, operator>> {
  list<string> values = names;
}

class AllElementCountsMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, ElementCount<"_self">.result,
                              "element count">;

class AllElementTypesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, ElementType<"_self">.result,
                              "element type">;

class AllRanksMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, Rank<"_self">.result, "rank">;

class AllShapesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, Shape<"_self">.result, "shape">;

class AllTypesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, "$_self.getType()", "type">;

// A type constraint that denotes `transform(lhs.getType()) == rhs.getType()`.
// An optional comparator function may be provided that changes the above form
// into: `comparator(transform(lhs.getType()), rhs.getType())`.
class TypesMatchWith<string summary, string lhsArg, string rhsArg,
                     string transform, string comparator = "std::equal_to<>()">
  : PredOpTrait<summary, CPred<
      comparator # "(" #
      !subst("$_self", "$" # lhsArg # ".getType()", transform) #
      ", $" # rhsArg # ".getType())">> {
  string lhs = lhsArg;
  string rhs = rhsArg;
  string transformer = transform;
}

// Special variant of `TypesMatchWith` that provides a comparator suitable for
// ranged arguments.
class RangedTypesMatchWith<string summary, string lhsArg, string rhsArg,
                           string transform>
  : TypesMatchWith<summary, lhsArg, rhsArg, transform, "llvm::equal">;

// Type Constraint operand `idx`'s Element type is `type`.
class TCopVTEtIs<int idx, Type type> : And<[
   CPred<"$_op.getNumOperands() > " # idx>,
   SubstLeaves<"$_self", "$_op.getOperand(" # idx # ").getType()",
     IsShapedTypePred>,
   SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(" # idx # "))",
     type.predicate>]>;

// Predicate to verify that a named argument or result's element type matches a
// given type.
class TypeIsPred<string name, Type type> :
   SubstLeaves<"$_self", "$" # name # ".getType()", type.predicate>;
class TypeIs<string name, Type type> : PredOpTrait<
  "'" # name # "' is " # type.summary, TypeIsPred<name, type>>;

// Predicate to verify that a named argument or result's element type matches a
// given type.
class ElementTypeIsPred<string name, Type type> : And<[
   SubstLeaves<"$_self", "$" # name # ".getType()", IsShapedTypePred>,
   SubstLeaves<"$_self", "getElementTypeOrSelf($" # name # ")",
     type.predicate>]>;
class ElementTypeIs<string name, Type type> : PredOpTrait<
  "'" # name # "' is " # type.summary, ElementTypeIsPred<name, type>>;

// Predicate to verify that the i'th operand and the j'th operand have the same
// elemental type.
// Type Constraint operand `i`'s Element type is Same As operand `j`'s Element
// type.
class TCopVTEtIsSameAs<int i, int j> : And<[
    CPred<"$_op.getNumOperands() > " # !if(!gt(i,j),i,j)>,
    SubstLeaves<"$_self", "$_op.getOperand(" # i # ").getType()",
      IsShapedTypePred>,
    SubstLeaves<"$_self", "$_op.getOperand(" # j # ").getType()",
      IsShapedTypePred>,
    CPred<"::mlir::getElementTypeOrSelf($_op.getOperand(" # i # ")) == "
          "::mlir::getElementTypeOrSelf($_op.getOperand(" # j # "))">]>;

// Predicate to verify that the i'th result and the j'th operand exist and has
// shaped types.
class TCOpResIsShapedTypePred<int i, int j> : And<[
    CPred<"$_op.getNumResults() > " # i>,
    CPred<"$_op.getNumOperands() > " # j>,
    SubstLeaves<"$_self", "$_op.getResult(" # i # ").getType()",
      IsShapedTypePred>,
    SubstLeaves<"$_self", "$_op.getOperand(" # j # ").getType()",
      IsShapedTypePred>]>;

// Predicate to verify that the i'th result and the j'th operand have the same
// type.
class TCresIsSameAsOpBase<int i, int j> :
    CPred<"$_op.getResult(" # i # ").getType() == "
          "$_op.getOperand(" # j # ").getType()">;

// Basic Predicate to verify that the i'th result and the j'th operand have the
// same elemental type.
class TCresVTEtIsSameAsOpBase<int i, int j> :
    CPred<"getElementTypeOrSelf($_op.getResult(" # i # ")) == "
          "getElementTypeOrSelf($_op.getOperand(" # j # "))">;

// Predicate to verify that the i'th result and the j'th operand have the same
// elemental type.
// Type Constraint result`i`'s Element type is Same As Operand `j`'s Element
// type.
class TCresVTEtIsSameAsOp<int i, int j> : And<[
    TCOpResIsShapedTypePred<i, j>,
    TCresVTEtIsSameAsOpBase<i, j>]>;

// Predicate to verify that the opId'th operand can be broadcasted to the type
// of the resId'th result.
class TCOpIsBroadcastableToRes<int opId, int resId> : And<[
    TCOpResIsShapedTypePred<opId, resId>,
    CPred<"::mlir::OpTrait::util::getBroadcastedType("
                  "$_op.getOperand(" # opId # ").getType(), "
                  "$_op.getResult(" # resId # ").getType())">]>;

// Predicate to verify that all the operands at the given `indices`
// have the same element type.
// Type Constraint operands' Element type are all Same At the given `indices`.
// We query the operands' types into a list and check they are all the same.
// Precondition:
// 1) all operands involved are of shaped type and
// 2) the indices are not out of range.
class TCopVTEtAreSameAt<list<int> indices> : CPred<
  "::llvm::all_equal(::llvm::map_range("
      "::mlir::ArrayRef<unsigned>({" # !interleave(indices, ", ") # "}), "
      "[this](unsigned i) { return getElementTypeOrSelf(this->getOperand(i)); "
      "}))">;

#endif // OP_BASE
