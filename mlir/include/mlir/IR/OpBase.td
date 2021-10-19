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
def IsVectorTypePred : CPred<"$_self.isa<::mlir::VectorType>()">;

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
// Dialect definitions
//===----------------------------------------------------------------------===//

// "Enum" values for emitAccessorPrefix of Dialect.
defvar kEmitAccessorPrefix_Raw = 0;      // Don't emit any getter/setter prefix.
defvar kEmitAccessorPrefix_Prefixed = 1; // Only emit with getter/setter prefix.
defvar kEmitAccessorPrefix_Both = 2;     // Emit without and with prefix.

class Dialect {
  // The name of the dialect.
  string name = ?;

  // Short summary of the dialect.
  string summary = ?;

  // The description of the dialect.
  string description = ?;

  // A list of dialects this dialect will load on construction as dependencies.
  // These are dialects that this dialect may involve in canonicalization
  // pattern or interfaces.
  list<string> dependentDialects = [];

  // The C++ namespace that ops of this dialect should be placed into.
  //
  // By default, uses the name of the dialect as the only namespace. To avoid
  // placing in any namespace, use "". To specify nested namespaces, use "::"
  // as the delimiter, e.g., given "A::B", ops will be placed in
  // `namespace A { namespace B { <ops> } }`.
  //
  // Note that this works in conjunction with dialect C++ code. Depending on how
  // the generated files are included into the dialect, you may want to specify
  // a full namespace path or a partial one.
  string cppNamespace = name;

  // An optional code block containing extra declarations to place in the
  // dialect declaration.
  code extraClassDeclaration = "";

  // If this dialect overrides the hook for materializing constants.
  bit hasConstantMaterializer = 0;

  /// If the dialect definition provides a non-default destructor.
  /// If false, a default destructor implementation will be generated.
  bit hasNonDefaultDestructor = 0;

  // If this dialect overrides the hook for verifying operation attributes.
  bit hasOperationAttrVerify = 0;

  // If this dialect overrides the hook for verifying region argument
  // attributes.
  bit hasRegionArgAttrVerify = 0;

  // If this dialect overrides the hook for verifying region result attributes.
  bit hasRegionResultAttrVerify = 0;

  // If this dialect overrides the hook for op interface fallback.
  bit hasOperationInterfaceFallback = 0;

  // If this dialect overrides the hook for canonicalization patterns.
  bit hasCanonicalizer = 0;

  // Whether to emit raw/with no prefix or format changes, or emit with
  // accessor with prefix only and UpperCamel suffix or to emit accessors with
  // both.
  //
  // If emitting with prefix is specified then the attribute/operand's
  // name is converted to UpperCamel from snake_case (which would result in
  // leaving UpperCamel unchanged while also converting lowerCamel to
  // UpperCamel) and prefixed with `get` or `set` depending on if it is a getter
  // or setter.
  int emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

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
    Type<t.predicate, summary> {
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
class Variadic<Type type> : TypeConstraint<type.predicate, type.summary> {
  Type baseType = type;
}

// A nested variadic type constraint. It expands to zero or more variadic ranges
// of the base type. This class is used for supporting variadic operands and
// results. `variadicSegmentAttrName` should correspond to the name of an
// I32ElementsAttr argument that provides the sizes of the inner variadic
// operand groups.
class VariadicOfVariadic<Type type, string variadicSegmentAttrName>
    : Variadic<type> {
  string segmentAttrName = variadicSegmentAttrName;
}

// An optional type constraint. It expands to either zero or one of the base
// type. This class is used for supporting optional operands/results.
class Optional<Type type> : TypeConstraint<type.predicate, type.summary> {
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
    // Satisfy any of the allowed type's condition
    Or<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypes, t.summary), " or "),
        summary),
    cppClassName>;

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

class Complex<Type type>
    : Type<And<[
          CPred<"$_self.isa<::mlir::ComplexType>()">,
          SubstLeaves<"$_self",
                      "$_self.cast<::mlir::ComplexType>().getElementType()",
           type.predicate>]>,
           "complex type with " # type.summary # " elements",
           "::mlir::ComplexType">,
      SameBuildabilityAs<type, "::mlir::ComplexType::get($_builder.get" # type #
                               "Type())"> {
  Type elementType = type;
}

def AnyComplex : Type<CPred<"$_self.isa<::mlir::ComplexType>()">,
                      "complex-type", "::mlir::ComplexType">;

class OpaqueType<string dialect, string name, string summary>
  : Type<CPred<"isOpaqueTypeWithName($_self, \""#dialect#"\", \""#name#"\")">,
         summary, "::mlir::OpaqueType">,
    BuildableType<"::mlir::OpaqueType::get("
                  "$_builder.getIdentifier(\"" # dialect # "\"), \""
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
         descr # " of " # etype.summary # " values", cppClassName> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypeCall = elementTypeCall;
}

class ShapedContainerType<list<Type> allowedTypes,
                          Pred containerPred, string descr,
                          string cppClassName = "::mlir::Type"> :
    ContainerType<AnyTypeOf<allowedTypes>, containerPred,
                  "$_self.cast<::mlir::ShapedType>().getElementType()", descr,
                  cppClassName>;

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

// Whether the number of elements of a vector is from the given
// `allowedRanks` list
class IsVectorOfRankPred<list<int> allowedRanks> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{$_self.cast<::mlir::VectorType>().getRank()
                           == }]
                         # allowedlength>)>]>;

// Any vector where the rank is from the given `allowedRanks` list
class VectorOfRank<list<int> allowedRanks> : Type<
  IsVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// Any vector where the rank is from the given `allowedRanks` list and the type
// is from the given `allowedTypes` list
class VectorOfRankAndType<list<int> allowedRanks,
                          list<Type> allowedTypes> : Type<
  And<[VectorOf<allowedTypes>.predicate,
       VectorOfRank<allowedRanks>.predicate]>,
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

// Any vector where the number of elements is from the given
// `allowedLengths` list
class VectorOfLength<list<int> allowedLengths> : Type<
  IsVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// Any vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes`
// list
class VectorOfLengthAndType<list<int> allowedLengths,
                            list<Type> allowedTypes> : Type<
  And<[VectorOf<allowedTypes>.predicate,
       VectorOfLength<allowedLengths>.predicate]>,
  VectorOf<allowedTypes>.summary # VectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

def AnyVector : VectorOf<[AnyType]>;

// Shaped types.

def AnyShaped: ShapedContainerType<[AnyType], IsShapedTypePred, "shaped",
                                   "::mlir::ShapedType">;

// Tensor types.

// Any tensor type whose element type is from the given `allowedTypes` list
class TensorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsTensorTypePred, "tensor",
                      "::mlir::TensorType">;

class RankedTensorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, And<[IsTensorTypePred, HasRankPred]>,
  "ranked tensor", "::mlir::TensorType">;

def AnyTensor : TensorOf<[AnyType]>;

// Unranked Memref type
class UnrankedTensorOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes,
                        IsUnrankedTensorTypePred,
                        "unranked.tensor", "::mlir::UnrankedTensorType">;

def AnyRankedTensor : RankedTensorOf<[AnyType]>;

// TODO: Have an easy way to add another constraint to a type.
class StaticShapeTensorOf<list<Type> allowedTypes>
    : Type<And<[TensorOf<allowedTypes>.predicate, HasStaticShapePred]>,
           "statically shaped " # TensorOf<allowedTypes>.summary,
           "::mlir::TensorType">;

def AnyStaticShapeTensor : StaticShapeTensorOf<[AnyType]>;

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

// Ranked tensor type with one of the specified types and ranks.
class TensorRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[TensorOf<allowedTypes>.predicate, HasAnyRankOfPred<ranks>]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         TensorOf<allowedTypes>.summary, "::mlir::TensorType">;

class 0DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [0]>;
class 1DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [1]>;
class 2DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [2]>;
class 3DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [3]>;
class 4DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [4]>;

// Unranked Memref type
class UnrankedMemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes,
                        IsUnrankedMemRefTypePred, "unranked.memref",
                        "::mlir::UnrankedMemRefType">;

def AnyUnrankedMemRef : UnrankedMemRefOf<[AnyType]>;

// Memref type.

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
    Type<And<[MemRefOf<allowedTypes>.predicate, HasAnyRankOfPred<ranks>]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

class StaticShapeMemRefOf<list<Type> allowedTypes>
    : Type<And<[MemRefOf<allowedTypes>.predicate, HasStaticShapePred]>,
           "statically shaped " # MemRefOf<allowedTypes>.summary,
           "::mlir::MemRefType">;

def AnyStaticShapeMemRef : StaticShapeMemRefOf<[AnyType]>;

// For a MemRefType, verify that it has strides.
def HasStridesPred : CPred<[{ isStrided($_self.cast<::mlir::MemRefType>()) }]>;

class StridedMemRefOf<list<Type> allowedTypes>
    : Type<And<[MemRefOf<allowedTypes>.predicate, HasStridesPred]>,
           "strided " # MemRefOf<allowedTypes>.summary>;

def AnyStridedMemRef : StridedMemRefOf<[AnyType]>;

class AnyStridedMemRefOfRank<int rank> :
  Type<And<[AnyStridedMemRef.predicate,
            MemRefRankOf<[AnyType], [rank]>.predicate]>,
       AnyStridedMemRef.summary # " of rank " # rank>;

class StridedMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate, HasAnyRankOfPred<ranks>]>,
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
                "::llvm::all_of(" # elementTypesCall # ", [](Type t) { return ",
                SubstLeaves<"$_self", "t", etype.predicate>,
                "; })"
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

// Type constraint for bool-like types: bools, vectors of bools, tensors of
// bools.
def BoolLike : TypeConstraint<Or<[I1.predicate, VectorOf<[I1]>.predicate,
                                  TensorOf<[I1]>.predicate]>,
    "bool-like">;

// Type constraint for signless-integer-like types: signless integers, indices,
// vectors of signless integers or indices, tensors of signless integers.
def SignlessIntegerLike : TypeConstraint<Or<[
        AnySignlessInteger.predicate, Index.predicate,
        VectorOf<[AnySignlessInteger, Index]>.predicate,
        TensorOf<[AnySignlessInteger, Index]>.predicate]>,
    "signless-integer-like">;

// Type constraint for float-like types: floats, vectors or tensors thereof.
def FloatLike : TypeConstraint<Or<[AnyFloat.predicate,
        VectorOf<[AnyFloat]>.predicate, TensorOf<[AnyFloat]>.predicate]>,
    "floating-point-like">;

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
class Attr<Pred condition, string descr = ""> :
    AttrConstraint<condition, descr> {
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
}

// An attribute of a specific dialect.
class DialectAttr<Dialect d, Pred condition, string descr = ""> :
    Attr<condition, descr> {
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

// Decorates an attribute as optional. The return type of the generated
// attribute accessor method will be Optional<>.
class OptionalAttr<Attr attr> : Attr<attr.predicate, attr.summary> {
  // Rewrite the attribute to be optional.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = "::llvm::Optional<" # attr.returnType #">";
  let convertFromStorage = "$_self ? " # returnType # "(" #
                           attr.convertFromStorage # ") : (::llvm::None)";
  let valueType = attr.valueType;
  let isOptional = 1;

  let baseAttr = attr;
}

// Default-valued string-based attribute. Wraps the default value in escaped
// quotes.
class DefaultValuedStrAttr<Attr attr, string val>
    : DefaultValuedAttr<attr, "\"" # val # "\"">;

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

// The mere presence of unit attributes has a meaning.  Therefore, unit
// attributes are always treated as optional and accessors to them return
// "true" if the attribute is present and "false" otherwise.
def UnitAttr : Attr<CPred<"$_self.isa<::mlir::UnitAttr>()">, "unit attribute"> {
  let storageType = [{ ::mlir::UnitAttr }];
  let constBuilderCall = "$_builder.getUnitAttr()";
  let convertFromStorage = "$_self != nullptr";
  let returnType = "bool";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// Enum attribute kinds

// Additional information for an enum attribute case.
class EnumAttrCaseInfo<string sym, int intVal, string strVal> {
  // The C++ enumerant symbol.
  string symbol = sym;

  // The C++ enumerant value.
  // If less than zero, there will be no explicit discriminator values assigned
  // to enumerators in the generated enum class.
  int value = intVal;

  // The string representation of the enumerant. May be the same as symbol.
  string str = strVal;
}

// An enum attribute case stored with StringAttr.
class StrEnumAttrCase<string sym, int val = -1> :
    EnumAttrCaseInfo<sym, val, sym>,
    StringBasedAttr<
      CPred<"$_self.cast<::mlir::StringAttr>().getValue() == \"" # sym # "\"">,
      "case " # sym>;

// An enum attribute case stored with IntegerAttr, which has an integer value,
// its representation as a string and a C++ symbol name which may be different.
class IntEnumAttrCaseBase<I intType, string sym, string strVal, int intVal> :
    EnumAttrCaseInfo<sym, intVal, strVal>,
    SignlessIntegerAttrBase<intType, "case " # strVal> {
  let predicate =
    CPred<"$_self.cast<::mlir::IntegerAttr>().getInt() == " # intVal>;
}

// Cases of integer enum attributes with a specific type. By default, the string
// representation is the same as the C++ symbol name.
class I32EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I32, sym, str, val>;
class I64EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I64, sym, str, val>;

// A bit enum case stored with 32-bit IntegerAttr. `val` here is *not* the
// ordinal number of the bit that is set. It is the 32-bit integer with only
// one bit set.
class BitEnumAttrCase<string sym, int val, string str = sym> :
    EnumAttrCaseInfo<sym, val, str>,
    SignlessIntegerAttrBase<I32, "case " # str> {
  let predicate = CPred<
    "$_self.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & "
    # val # "u">;
}

// Additional information for an enum attribute.
class EnumAttrInfo<
    string name, list<EnumAttrCaseInfo> cases, Attr baseClass> :
      Attr<baseClass.predicate, baseClass.summary> {
  // The C++ enum class name
  string className = name;

  // List of all accepted cases
  list<EnumAttrCaseInfo> enumerants = cases;

  // The following fields are only used by the EnumsGen backend to generate
  // an enum class definition and conversion utility functions.

  // The underlying type for the C++ enum class. An empty string mean the
  // underlying type is not explicitly specified.
  string underlyingType = "";

  // The name of the utility function that converts a value of the underlying
  // type to the corresponding symbol. It will have the following signature:
  //
  // ```c++
  // llvm::Optional<<qualified-enum-class-name>> <fn-name>(<underlying-type>);
  // ```
  string underlyingToSymbolFnName = "symbolize" # name;

  // The name of the utility function that converts a string to the
  // corresponding symbol. It will have the following signature:
  //
  // ```c++
  // llvm::Optional<<qualified-enum-class-name>> <fn-name>(llvm::StringRef);
  // ```
  string stringToSymbolFnName = "symbolize" # name;

  // The name of the utility function that converts a symbol to the
  // corresponding string. It will have the following signature:
  //
  // ```c++
  // <return-type> <fn-name>(<qualified-enum-class-name>);
  // ```
  string symbolToStringFnName = "stringify" # name;
  string symbolToStringFnRetType = "::llvm::StringRef";

  // The name of the utility function that returns the max enum value used
  // within the enum class. It will have the following signature:
  //
  // ```c++
  // static constexpr unsigned <fn-name>();
  // ```
  string maxEnumValFnName = "getMaxEnumValFor" # name;

  // Generate specialized Attribute class
  bit genSpecializedAttr = 1;
  // The underlying Attribute class, which holds the enum value
  Attr baseAttrClass = baseClass;
  // The name of specialized Enum Attribute class
  string specializedAttrClassName = name # Attr;

  // Override Attr class fields for specialized class
  let predicate = !if(genSpecializedAttr,
    CPred<"$_self.isa<" # cppNamespace # "::" # specializedAttrClassName # ">()">,
    baseAttrClass.predicate);
  let storageType = !if(genSpecializedAttr,
    cppNamespace # "::" # specializedAttrClassName,
    baseAttrClass.storageType);
  let returnType = !if(genSpecializedAttr,
    cppNamespace # "::" # className,
    baseAttrClass.returnType);
  let constBuilderCall = !if(genSpecializedAttr,
    cppNamespace # "::" # specializedAttrClassName # "::get($_builder.getContext(), $0)",
    baseAttrClass.constBuilderCall);
  let valueType = baseAttrClass.valueType;
}

// An enum attribute backed by StringAttr.
//
// Op attributes of this kind are stored as StringAttr. Extra verification will
// be generated on the string though: only the symbols of the allowed cases are
// permitted as the string value.
class StrEnumAttr<string name, string summary, list<StrEnumAttrCase> cases> :
  EnumAttrInfo<name, cases,
    StringBasedAttr<
      And<[StrAttr.predicate, Or<!foreach(case, cases, case.predicate)>]>,
      !if(!empty(summary), "allowed string cases: " #
          !interleave(!foreach(case, cases, "'" # case.symbol # "'"), ", "),
          summary)>> {
  // Disable specialized Attribute class for `StringAttr` backend by default.
  let genSpecializedAttr = 0;
}

// An enum attribute backed by IntegerAttr.
//
// Op attributes of this kind are stored as IntegerAttr. Extra verification will
// be generated on the integer though: only the values of the allowed cases are
// permitted as the integer value.
class IntEnumAttrBase<I intType, list<IntEnumAttrCaseBase> cases, string summary> :
    SignlessIntegerAttrBase<intType, summary> {
  let predicate = And<[
    SignlessIntegerAttrBase<intType, summary>.predicate,
    Or<!foreach(case, cases, case.predicate)>]>;
}

class IntEnumAttr<I intType, string name, string summary,
                  list<IntEnumAttrCaseBase> cases> :
  EnumAttrInfo<name, cases,
    IntEnumAttrBase<intType, cases,
      !if(!empty(summary), "allowed " # intType.summary # " cases: " #
          !interleave(!foreach(case, cases, case.value), ", "),
          summary)>>;

class I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases> :
    IntEnumAttr<I32, name, summary, cases> {
  let underlyingType = "uint32_t";
}
class I64EnumAttr<string name, string summary, list<I64EnumAttrCase> cases> :
    IntEnumAttr<I64, name, summary, cases> {
  let underlyingType = "uint64_t";
}

// A bit enum stored with 32-bit IntegerAttr.
//
// Op attributes of this kind are stored as IntegerAttr. Extra verification will
// be generated on the integer to make sure only allowed bit are set. Besides,
// helper methods are generated to parse a string separated with a specified
// delimiter to a symbol and vice versa.
class BitEnumAttrBase<list<BitEnumAttrCase> cases, string summary> :
    SignlessIntegerAttrBase<I32, summary> {
  let predicate = And<[
    I32Attr.predicate,
    // Make sure we don't have unknown bit set.
    CPred<"!($_self.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~("
          # !interleave(!foreach(case, cases, case.value # "u"), "|") #
          ")))">
  ]>;
}

class BitEnumAttr<string name, string summary, list<BitEnumAttrCase> cases> :
    EnumAttrInfo<name, cases, BitEnumAttrBase<cases, summary>> {
  let underlyingType = "uint32_t";

  // We need to return a string because we may concatenate symbols for multiple
  // bits together.
  let symbolToStringFnRetType = "std::string";

  // The delimiter used to separate bit enum cases in strings.
  string separator = "|";
}

//===----------------------------------------------------------------------===//
// Composite attribute kinds

class DictionaryAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::DictionaryAttr }];
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
    "::llvm::makeArrayRef($0)).cast<::mlir::DenseIntElementsAttr>()";
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
    "}, $_builder.getIntegerType(" # width # ")), ::llvm::makeArrayRef($0))";
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
    "::llvm::makeArrayRef($0))";
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
    "::llvm::makeArrayRef($0)).cast<::mlir::DenseFPElementsAttr>()";
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
                            "[&](::mlir::Attribute attr) { return ",
                               SubstLeaves<"$_self", "attr", element.predicate>,
                            "; })">]>,
    summary> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";

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

// Attribute information for an Attribute field within a StructAttr.
class StructFieldAttr<string thisName, Attr thisType> {
  // Name of this field in the StructAttr.
  string name = thisName;

  // Attribute type wrapped by the struct attr.
  Attr type = thisType;
}

// Structured attribute that wraps a DictionaryAttr and provides both a
// validation method and set of accessors for a fixed set of fields. This is
// useful when representing data that would normally be in a structure.
class StructAttr<string name, Dialect d,
                 list<StructFieldAttr> attributes> :
    DictionaryAttrBase<CPred<"$_self.isa<" # d.cppNamespace
                                           # "::" # name # ">()">,
        "DictionaryAttr with field(s): " #
        !interleave(!foreach(a, attributes, "'" # a.name # "'"), ", ") #
        " (each field having its own constraints)"> {
  // Name for this StructAttr.
  string className = name;

  // Return type should match the name of the structure.
  let returnType = d.cppNamespace # "::" # name;

  // Storage type should match the name of the structure.
  let storageType = d.cppNamespace # "::" # name;

  // The dialect this StructAttr belongs to.
  Dialect dialect = d;

  let cppNamespace = d.cppNamespace;

  // List of fields that the StructAttr contains.
  list<StructFieldAttr> fields = attributes;
}

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
  // * `$_ctx` will be replaced by the MLIRContext* instance.
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
def ConstUnitAttr : ConstantAttr<UnitAttr, "unit">;

// Constant string-based attribute. Wraps the desired string in escaped quotes.
class ConstantStrAttr<Attr attribute, string val>
    : ConstantAttr<attribute, "\"" # val # "\"">;

//===----------------------------------------------------------------------===//
// Common attribute constraints
//===----------------------------------------------------------------------===//

// A general mechanism to further confine the given `attr` with all the
// `constraints`. This allows to compose complex constraints out of a series
// of more primitive ones.
class Confined<Attr attr, list<AttrConstraint> constraints> : Attr<
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
class AllAttrConstraintsOf<list<AttrConstraint> constraints> : AttrConstraint<
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
    CPred<"$_self.cast<IntegerAttr>().getValue().isStrictlyPositive()">,
    "whose value is positive">;

class ArrayMinCount<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::ArrayAttr>().size() >= " # n>,
    "with at least " # n # " elements">;

class ArrayCount<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::ArrayAttr>().size() == " #n>,
    "with exactly " # n # " elements">;

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

// An attribute constraint on FlatSymbolRefAttr that requires that the
// reference point to an op of `opClass` within the closest parent with a symbol
// table.
// TODO: Add support for nested symbol references.
class ReferToOp<string opClass> : AttrConstraint<
    CPred<"isa_and_nonnull<" # opClass # ">("
            "::mlir::SymbolTable::lookupNearestSymbolFrom("
              "&$_op, $_self.cast<::mlir::FlatSymbolRefAttr>().getAttr()))">,
    "referencing to a '" # opClass # "' symbol">;

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
// TypeTrait definitions
//===----------------------------------------------------------------------===//

// TypeTrait represents a trait regarding a type.
// TODO: Remove this class in favor of using Trait.
class TypeTrait;

// These classes are used to define type specific traits.
class NativeTypeTrait<string name> : NativeTrait<name, "Type">, TypeTrait;
class ParamNativeTypeTrait<string prop, string params>
    : ParamNativeTrait<prop, params, "Type">, TypeTrait;
class GenInternalTypeTrait<string prop>
    : GenInternalTrait<prop, "Type">, TypeTrait;
class PredTypeTrait<string descr, Pred pred>
    : PredTrait<descr, pred>, TypeTrait;

//===----------------------------------------------------------------------===//
// AttrTrait definitions
//===----------------------------------------------------------------------===//

// AttrTrait represents a trait regarding an attribute.
// TODO: Remove this class in favor of using Trait.
class AttrTrait;

// These classes are used to define attribute specific traits.
class NativeAttrTrait<string name> : NativeTrait<name, "Attribute">, AttrTrait;
class ParamNativeAttrTrait<string prop, string params>
    : ParamNativeTrait<prop, params, "Attribute">, AttrTrait;
class GenInternalAttrTrait<string prop>
    : GenInternalTrait<prop, "Attribute">, AttrTrait;
class PredAttrTrait<string descr, Pred pred>
    : PredTrait<descr, pred>, AttrTrait;

//===----------------------------------------------------------------------===//
// OpTrait definitions
//===----------------------------------------------------------------------===//

// OpTrait represents a trait regarding an operation.
// TODO: Remove this class in favor of using Trait.
class OpTrait;

// Define a OpTrait corresponding to a list of OpTraits, this allows for
// specifying a list of traits as trait. Avoids needing to do
// `[Traits, ...] # ListOfTraits # [Others, ...]` while still allowing providing
// convenient groupings.
class OpTraitList<list<OpTrait> props> : OpTrait {
  list<OpTrait> traits = props;
}

// These classes are used to define operation specific traits.
class NativeOpTrait<string name> : NativeTrait<name, "Op">, OpTrait;
class ParamNativeOpTrait<string prop, string params>
    : ParamNativeTrait<prop, params, "Op">, OpTrait;
class GenInternalOpTrait<string prop> : GenInternalTrait<prop, "Op">, OpTrait;
class PredOpTrait<string descr, Pred pred> : PredTrait<descr, pred>, OpTrait;

// Op defines an affine scope.
def AffineScope : NativeOpTrait<"AffineScope">;
// Op defines an automatic allocation scope.
def AutomaticAllocationScope : NativeOpTrait<"AutomaticAllocationScope">;
// Op supports operand broadcast behavior.
def ResultsBroadcastableShape :
  NativeOpTrait<"ResultsBroadcastableShape">;
// X op Y == Y op X
def Commutative  : NativeOpTrait<"IsCommutative">;
// op op X == op X
def Idempotent  : NativeOpTrait<"IsIdempotent">;
// op op X == X
def Involution  : NativeOpTrait<"IsInvolution">;
// Op behaves like a constant.
def ConstantLike : NativeOpTrait<"ConstantLike">;
// Op behaves like a function.
def FunctionLike : NativeOpTrait<"FunctionLike">;
// Op is isolated from above.
def IsolatedFromAbove : NativeOpTrait<"IsIsolatedFromAbove">;
// Op results are float or vectors/tensors thereof.
def ResultsAreFloatLike : NativeOpTrait<"ResultsAreFloatLike">;
// Op has the same operand type.
def SameTypeOperands : NativeOpTrait<"SameTypeOperands">;
// Op has same shape for all operands.
def SameOperandsShape : NativeOpTrait<"SameOperandsShape">;
// Op has same operand and result shape.
def SameOperandsAndResultShape : NativeOpTrait<"SameOperandsAndResultShape">;
// Op has the same operand and result type.
def SameOperandsAndResultType : NativeOpTrait<"SameOperandsAndResultType">;
// Op has the same element type (or type itself, if scalar) for all operands.
def SameOperandsElementType : NativeOpTrait<"SameOperandsElementType">;
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
def Scalarizable : NativeOpTrait<"Scalarizable">;
// Elementwise op can be applied to all-vector operands.
def Vectorizable : NativeOpTrait<"Vectorizable">;
// Elementwise op can be applied to all-tensor operands.
def Tensorizable : NativeOpTrait<"Tensorizable">;

// Group together `Elementwise`, `Scalarizable`, `Vectorizable`, and
// `Tensorizable` for convenience.
def ElementwiseMappable {
  list<OpTrait> traits = [
    Elementwise,
    Scalarizable,
    Vectorizable,
    Tensorizable,
  ];
}

// Op's regions have a single block.
def SingleBlock : NativeOpTrait<"SingleBlock">;

// Op's regions have a single block with the specified terminator.
class SingleBlockImplicitTerminator<string op>
    : ParamNativeOpTrait<"SingleBlockImplicitTerminator", op>;

// Op's regions don't have terminator.
def NoTerminator : NativeOpTrait<"NoTerminator">;

// Op's parent operation is the provided one.
class HasParent<string op>
    : ParamNativeOpTrait<"HasParent", op>;

class ParentOneOf<list<string> ops>
    : ParamNativeOpTrait<"HasParent", !interleave(ops, ", ")>;

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
def AttrSizedOperandSegments : NativeOpTrait<"AttrSizedOperandSegments">;
// Similar to AttrSizedOperandSegments, but used for results. The attribute
// should be named as `result_segment_sizes`.
def AttrSizedResultSegments  : NativeOpTrait<"AttrSizedResultSegments">;

// Op attached regions have no arguments
def NoRegionArguments : NativeOpTrait<"NoRegionArguments">;

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
class OpInterfaceTrait<string name, code verifyBody = [{}]>
    : InterfaceTrait<name>, OpTrait {
  // Specify the body of the verification function. `$_op` will be replaced with
  // the operation being verified.
  code verify = verifyBody;
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
  string cppClassName = name;

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
}

// AttrInterface represents an interface registered to an attribute.
class AttrInterface<string name> : Interface<name>, InterfaceTrait<name>;

// OpInterface represents an interface registered to an operation.
class OpInterface<string name> : Interface<name>, OpInterfaceTrait<name>;

// TypeInterface represents an interface registered to a type.
class TypeInterface<string name> : Interface<name>, InterfaceTrait<name>;

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
        AttrInterface<interface.cppClassName> {
    let description = interface.description;
    let cppClassName = interface.cppClassName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
}
class DeclareOpInterfaceMethods<OpInterface interface,
                                list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        OpInterface<interface.cppClassName> {
    let description = interface.description;
    let cppClassName = interface.cppClassName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
}
class DeclareTypeInterfaceMethods<TypeInterface interface,
                                  list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        TypeInterface<interface.cppClassName> {
    let description = interface.description;
    let cppClassName = interface.cppClassName;
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
class Op<Dialect dialect, string mnemonic, list<OpTrait> props = []> {
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

  // Custom parser.
  code parser = ?;

  // Custom printer.
  code printer = ?;

  // Custom assembly format.
  string assemblyFormat = ?;

  // Custom verifier.
  code verifier = ?;

  // Whether this op has associated canonicalization patterns.
  bit hasCanonicalizer = 0;

  // Whether this op has a static "canonicalize" method to perform "match and
  // rewrite patterns".
  bit hasCanonicalizeMethod = 0;

  // Whether this op has a folder.
  bit hasFolder = 0;

  // Op traits.
  // Note: The list of traits will be uniqued by ODS.
  list<OpTrait> traits = props;

  // Additional code that will be added to the public part of the generated
  // C++ code of the op declaration.
  code extraClassDeclaration = ?;
}

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class BaseOpWithOffsetSizesAndStrides<Dialect dialect, string mnemonic,
                                      list<OpTrait> traits = []> :
  Op<dialect, mnemonic, traits> {

  // For every such op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  code extraBaseClassDeclaration = [{
    /// Returns the dynamic sizes for this subview operation if specified.
    operand_range getDynamicSizes() { return sizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    SmallVector<Range, 8> getOrCreateRanges(OpBuilder &b, Location loc) {
      return mlir::getOrCreateRanges(*this, b, loc);
    }
  }];
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
// Common value constraints
//===----------------------------------------------------------------------===//

def HasNoUseOf: Constraint<
    CPred<"$_self.use_empty()">, "has no use">;

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
    CPred<"::llvm::is_splat(::llvm::makeArrayRef({"
          # !interleave(values, ", ") #"}))">;

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
  "::llvm::is_splat(::llvm::map_range("
      "::mlir::ArrayRef<unsigned>({" # !interleave(indices, ", ") # "}), "
      "[this](unsigned i) { return getElementTypeOrSelf(this->getOperand(i)); "
      "}))">;

//===----------------------------------------------------------------------===//
// Pattern definitions
//===----------------------------------------------------------------------===//

// Marker used to identify the delta value added to the default benefit value.
def addBenefit;

// Base class for op+ -> op+ rewrite rules. These allow declaratively
// specifying rewrite rules.
//
// A rewrite rule contains two components: a source pattern and one or more
// result patterns. Each pattern is specified as a (recursive) DAG node (tree)
// in the form of `(node arg0, arg1, ...)`.
//
// The `node` are normally MLIR ops, but it can also be one of the directives
// listed later in this section.
//
// ## Symbol binding
//
// In the source pattern, `argN` can be used to specify matchers (e.g., using
// type/attribute type constraints, etc.) and bound to a name for later use.
// We can also bind names to op instances to reference them later in
// multi-entity constraints. Operands in the source pattern can have
// the same name. This bounds one operand to the name while verifying
// the rest are all equal.
//
//
// In the result pattern, `argN` can be used to refer to a previously bound
// name, with potential transformations (e.g., using tAttr, etc.). `argN` can
// itself be nested DAG node. We can also bound names to ops to reference
// them later in other result patterns.
//
// For example,
//
// ```
// def : Pattern<(OneResultOp1:$op1 $arg0, $arg1, $arg0),
//               [(OneResultOp2:$op2 $arg0, $arg1),
//                (OneResultOp3 $op2 (OneResultOp4))],
//               [(HasStaticShapePred $op1)]>;
// ```
//
// First `$arg0` and '$arg1' are bound to the `OneResultOp1`'s first
// and second arguments and used later to build `OneResultOp2`. Second '$arg0'
// is verified to be equal to the first '$arg0' operand.
// `$op1` is bound to `OneResultOp1` and used to check whether the result's
// shape is static. `$op2` is bound to `OneResultOp2` and used to
// build `OneResultOp3`.
//
// ## Multi-result op
//
// To create multi-result ops in result pattern, you can use a syntax similar
// to uni-result op, and it will act as a value pack for all results:
//
// ```
// def : Pattern<(ThreeResultOp ...),
//               [(TwoResultOp ...), (OneResultOp ...)]>;
// ```
//
// Then `TwoResultOp` will replace the first two values of `ThreeResultOp`.
//
// You can also use `$<name>__N` to explicitly access the N-th result.
// ```
// def : Pattern<(FiveResultOp ...),
//               [(TwoResultOp1:$res1__1 ...), (replaceWithValue $res1__0),
//                (TwoResultOp2:$res2 ...), (replaceWithValue $res2__1)]>;
// ```
//
// Then the values generated by `FiveResultOp` will be replaced by
//
// * `FiveResultOp`#0: `TwoResultOp1`#1
// * `FiveResultOp`#1: `TwoResultOp1`#0
// * `FiveResultOp`#2: `TwoResultOp2`#0
// * `FiveResultOp`#3: `TwoResultOp2`#1
// * `FiveResultOp`#4: `TwoResultOp2`#1
class Pattern<dag source, list<dag> results, list<dag> preds = [],
  dag benefitAdded = (addBenefit 0)> {
  dag sourcePattern = source;
  // Result patterns. Each result pattern is expected to replace one result
  // of the root op in the source pattern. In the case of more result patterns
  // than needed to replace the source op, only the last N results generated
  // by the last N result pattern is used to replace a N-result source op.
  // So that the beginning result patterns can be used to generate additional
  // ops to aid building the results used for replacement.
  list<dag> resultPatterns = results;
  // Multi-entity constraints. Each constraint here involves multiple entities
  // matched in source pattern and places further constraints on them as a
  // whole.
  list<dag> constraints = preds;
  // The delta value added to the default benefit value. The default value is
  // the number of ops in the source pattern. The rule with the highest final
  // benefit value will be applied first if there are multiple rules matches.
  // This delta value can be either positive or negative.
  dag benefitDelta = benefitAdded;
}

// Form of a pattern which produces a single result.
class Pat<dag pattern, dag result, list<dag> preds = [],
  dag benefitAdded = (addBenefit 0)> :
  Pattern<pattern, [result], preds, benefitAdded>;

// Native code call wrapper. This allows invoking an arbitrary C++ expression
// to create an op operand/attribute or replace an op result.
//
// ## Placeholders
//
// If used as a DAG leaf, i.e., `(... NativeCodeCall<"...">:$arg, ...)`,
// the wrapped expression can take special placeholders listed below:
//
// * `$_builder` will be replaced by the current `mlir::PatternRewriter`.
// * `$_self` will be replaced by the defining operation in a source pattern.
//   E.g., `NativeCodeCall<"Foo($_self, &$0)> I32Attr:$attr)>`, `$_self` will be
//   replaced with the defining operation of the first operand of OneArgOp.
//
// If used as a DAG node, i.e., `(NativeCodeCall<"..."> <arg0>, ..., <argN>)`,
// then positional placeholders are also supported; placeholder `$N` in the
// wrapped C++ expression will be replaced by `<argN>`.
//
// ## Bind multiple results
//
// To bind multi-results and access the N-th result with `$<name>__N`, specify
// the number of return values in the template. Note that only `Value` type is
// supported for multiple results binding.

class NativeCodeCall<string expr, int returns = 1> {
  string expression = expr;
  int numReturns = returns;
}

class NativeCodeCallVoid<string expr> : NativeCodeCall<expr, 0>;

def ConstantLikeMatcher : NativeCodeCall<"success(matchPattern($_self->getResult(0), m_Constant(&$0)))">;

//===----------------------------------------------------------------------===//
// Rewrite directives
//===----------------------------------------------------------------------===//

// Directive used in result pattern to indicate that no new op are generated,
// so to replace the matched DAG with an existing SSA value.
def replaceWithValue;

// Directive used in result patterns to specify the location of the generated
// op. This directive must be used as a trailing argument to op creation or
// native code calls.
//
// Usage:
// * Create a named location: `(location "myLocation")`
// * Copy the location of a captured symbol: `(location $arg)`
// * Create a fused location: `(location "metadata", $arg0, $arg1)`

def location;

// Directive used in result patterns to specify return types for a created op.
// This allows ops to be created without relying on type inference with
// `OpTraits` or an op builder with deduction.
//
// This directive must be used as a trailing argument to op creation.
//
// Specify one return type with a string literal:
//
// ```
// (AnOp $val, (returnType "$_builder.getI32Type()"))
// ```
//
// Pass a captured value to copy its return type:
//
// ```
// (AnOp $val, (returnType $val));
// ```
//
// Pass a native code call inside a DAG to create a new type with arguments.
//
// ```
// (AnOp $val,
//       (returnType (NativeCodeCall<"$_builder.getTupleType({$0})"> $val)));
// ```
//
// Specify multiple return types with multiple of any of the above.

def returnType;

//===----------------------------------------------------------------------===//
// Attribute and Type generation
//===----------------------------------------------------------------------===//

// Class for defining a custom getter.
//
// TableGen generates several generic getter methods for each attribute and type
// by default, corresponding to the specified dag parameters. If the default
// generated ones cannot cover some use case, custom getters can be defined
// using instances of this class.
//
// The signature of the `get` is always either:
//
// ```c++
// static <ClassName> get(MLIRContext *context, <other-parameters>...) {
//   <body>...
// }
// ```
//
// or:
//
// ```c++
// static <ClassName> get(MLIRContext *context, <parameters>...);
// ```
//
// To define a custom getter, the parameter list and body should be passed
// in as separate template arguments to this class. The parameter list is a
// TableGen DAG with `ins` operation with named arguments, which has either:
//   - string initializers ("Type":$name) to represent a typed parameter, or
//   - CArg-typed initializers (CArg<"Type", "default">:$name) to represent a
//     typed parameter that may have a default value.
// The type string is used verbatim to produce code and, therefore, must be a
// valid C++ type. It is used inside the C++ namespace of the parent Type's
// dialect; explicit namespace qualification like `::mlir` may be necessary if
// Types are not placed inside the `mlir` namespace. The default value string is
// used verbatim to produce code and must be a valid C++ initializer the given
// type. For example, the following signature specification
//
// ```
// AttrOrTypeBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg)>
// ```
//
// has an integer parameter and a float parameter with a default value.
//
// If an empty string is passed in for `body`, then *only* the builder
// declaration will be generated; this provides a way to define complicated
// builders entirely in C++. If a `body` string is provided, the `Base::get`
// method should be invoked using `$_get`, e.g.:
//
// ```
// AttrOrTypeBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg), [{
//   return $_get($_ctxt, integerArg, floatArg);
// }]>
// ```
//
// This is necessary because the `body` is also used to generate `getChecked`
// methods, which have a different underlying `Base::get*` call.
//
class AttrOrTypeBuilder<dag parameters, code bodyCode = ""> {
  dag dagParams = parameters;
  code body = bodyCode;

  // The context parameter can be inferred from one of the other parameters and
  // is not implicitly added to the parameter list.
  bit hasInferredContextParam = 0;
}
class AttrBuilder<dag parameters, code bodyCode = "">
  : AttrOrTypeBuilder<parameters, bodyCode>;
class TypeBuilder<dag parameters, code bodyCode = "">
  : AttrOrTypeBuilder<parameters, bodyCode>;

// A class of AttrOrTypeBuilder that is able to infer the MLIRContext parameter
// from one of the other builder parameters. Instances of this builder do not
// have `MLIRContext *` implicitly added to the parameter list.
class AttrOrTypeBuilderWithInferredContext<dag parameters, code bodyCode = "">
  : TypeBuilder<parameters, bodyCode> {
  let hasInferredContextParam = 1;
}
class AttrBuilderWithInferredContext<dag parameters, code bodyCode = "">
  : AttrOrTypeBuilderWithInferredContext<parameters, bodyCode>;
class TypeBuilderWithInferredContext<dag parameters, code bodyCode = "">
  : AttrOrTypeBuilderWithInferredContext<parameters, bodyCode>;

// Define a new attribute or type, named `name`, that inherits from the given
// C++ base class.
class AttrOrTypeDef<string valueType, string name, list<Trait> defTraits,
                    string baseCppClass> {
  // The name of the C++ base class to use for this def.
  string cppBaseClassName = baseCppClass;

  // Additional, longer human-readable description of what the def does.
  string description = "";

  // Name of storage class to generate or use.
  string storageClass = name # valueType # "Storage";

  // Namespace (withing dialect c++ namespace) in which the storage class
  // resides.
  string storageNamespace = "detail";

  // Specify if the storage class is to be generated.
  bit genStorageClass = 1;

  // Specify that the generated storage class has a constructor which is written
  // in C++.
  bit hasStorageCustomConstructor = 0;

  // The list of parameters for this type. Parameters will become both
  // parameters to the get() method and storage class member variables.
  //
  // The format of this dag is:
  //    (ins
  //        "<c++ type>":$param1Name,
  //        "<c++ type>":$param2Name,
  //        AttrOrTypeParameter<"c++ type", "param description">:$param3Name)
  // AttrOrTypeParameters (or more likely one of their subclasses) are required
  // to add more information about the parameter, specifically:
  //  - Documentation
  //  - Code to allocate the parameter (if allocation is needed in the storage
  //    class constructor)
  //
  // For example:
  //    (ins "int":$width,
  //         ArrayRefParameter<"bool", "list of bools">:$yesNoArray)
  //
  // (ArrayRefParameter is a subclass of AttrOrTypeParameter which has
  // allocation code for re-allocating ArrayRefs. It is defined below.)
  dag parameters = (ins);

  // Custom builder methods.
  // In addition to the custom builders provided here, and unless
  // skipDefaultBuilders is set, a default builder is generated with the
  // following signature:
  //
  // ```c++
  // static <ClassName> get(MLIRContext *, <parameters>);
  // ```
  //
  // Note that builders should only be provided when a def has parameters.
  list<AttrOrTypeBuilder> builders = ?;

  // The list of traits attached to this def.
  list<Trait> traits = defTraits;

  // Use the lowercased name as the keyword for parsing/printing. Specify only
  // if you want tblgen to generate declarations and/or definitions of
  // the printer/parser.
  string mnemonic = ?;

  // If 'mnemonic' specified,
  //   If null, generate just the declarations.
  //   If a non-empty code block, just use that code as the definition code.
  //   Error if an empty code block.
  code printer = ?;
  code parser = ?;

  // If set, generate accessors for each parameter.
  bit genAccessors = 1;

  // Avoid generating default get/getChecked functions. Custom get methods must
  // be provided.
  bit skipDefaultBuilders = 0;

  // Generate the verify and getChecked methods.
  bit genVerifyDecl = 0;

  // Extra code to include in the class declaration.
  code extraClassDeclaration = [{}];
}

// Define a new attribute, named `name`, belonging to `dialect` that inherits
// from the given C++ base class.
class AttrDef<Dialect dialect, string name, list<Trait> traits = [],
              string baseCppClass = "::mlir::Attribute">
    : DialectAttr<dialect, CPred<"">, /*descr*/"">,
      AttrOrTypeDef<"Attr", name, traits, baseCppClass> {
  // The name of the C++ Attribute class.
  string cppClassName = name # "Attr";
  let storageType = dialect.cppNamespace # "::" # name # "Attr";

  // The underlying C++ value type
  let returnType = dialect.cppNamespace # "::" # cppClassName;

  // The call expression to convert from the storage type to the return
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format: $_self will be expanded to the attribute.
  //
  // For example, `$_self.getValue().getSExtValue()` for `IntegerAttr val` will
  // expand to `getAttrOfType<IntegerAttr>("val").getValue().getSExtValue()`.
  let convertFromStorage = "$_self.cast<" # dialect.cppNamespace #
                                 "::" # cppClassName # ">()";

  // A code block used to build the value 'Type' of an Attribute when
  // initializing its storage instance. This field is optional, and if not
  // present the attribute will have its value type set to `NoneType`. This code
  // block may reference any of the attributes parameters via
  // `$_<parameter-name`. If one of the parameters of the attribute is of type
  // `AttributeSelfTypeParameter`, this field is ignored.
  code typeBuilder = ?;

  // The predicate for when this def is used as a constraint.
  let predicate = CPred<"$_self.isa<" # dialect.cppNamespace #
                                 "::" # cppClassName # ">()">;
}

// Define a new type, named `name`, belonging to `dialect` that inherits from
// the given C++ base class.
class TypeDef<Dialect dialect, string name, list<Trait> traits = [],
              string baseCppClass = "::mlir::Type">
    : DialectType<dialect, CPred<"">, /*descr*/"", name # "Type">,
      AttrOrTypeDef<"Type", name, traits, baseCppClass> {
  // A constant builder provided when the type has no parameters.
  let builderCall = !if(!empty(parameters),
                           "$_builder.getType<" # dialect.cppNamespace #
                               "::" # cppClassName # ">()",
                           "");
  // The predicate for when this def is used as a constraint.
  let predicate = CPred<"$_self.isa<" # dialect.cppNamespace #
                                 "::" # cppClassName # ">()">;
}

// 'Parameters' should be subclasses of this or simple strings (which is a
// shorthand for AttrOrTypeParameter<"C++Type">).
class AttrOrTypeParameter<string type, string desc, string accessorType = ""> {
  // Custom memory allocation code for storage constructor.
  code allocator = ?;
  // Custom comparator used to compare two instances for equality.
  code comparator = ?;
  // The C++ type of this parameter.
  string cppType = type;
  // The C++ type of the accessor for this parameter.
  string cppAccessorType = !if(!empty(accessorType), type, accessorType);
  // One-line human-readable description of the argument.
  string summary = desc;
  // The format string for the asm syntax (documentation only).
  string syntax = ?;
}
class AttrParameter<string type, string desc, string accessorType = "">
 : AttrOrTypeParameter<type, desc, accessorType>;
class TypeParameter<string type, string desc, string accessorType = "">
 : AttrOrTypeParameter<type, desc, accessorType>;

// For StringRefs, which require allocation.
class StringRefParameter<string desc = ""> :
    AttrOrTypeParameter<"::llvm::StringRef", desc> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
}

// For APFloats, which require comparison.
class APFloatParameter<string desc> :
    AttrOrTypeParameter<"::llvm::APFloat", desc> {
  let comparator = "$_lhs.bitwiseIsEqual($_rhs)";
}

// For standard ArrayRefs, which require allocation.
class ArrayRefParameter<string arrayOf, string desc = ""> :
    AttrOrTypeParameter<"::llvm::ArrayRef<" # arrayOf # ">", desc> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
}

// For classes which require allocation and have their own allocateInto method.
class SelfAllocationParameter<string type, string desc> :
    AttrOrTypeParameter<type, desc> {
  let allocator = [{$_dst = $_self.allocateInto($_allocator);}];
}

// For ArrayRefs which contain things which allocate themselves.
class ArrayRefOfSelfAllocationParameter<string arrayOf, string desc> :
    AttrOrTypeParameter<"::llvm::ArrayRef<" # arrayOf # ">", desc> {
  let allocator = [{
    llvm::SmallVector<}] # arrayOf # [{, 4> tmpFields;
    for (size_t i = 0, e = $_self.size(); i < e; ++i)
      tmpFields.push_back($_self[i].allocateInto($_allocator));
    $_dst = $_allocator.copyInto(ArrayRef<}] # arrayOf # [{>(tmpFields));
  }];
}

// This is a special parameter used for AttrDefs that represents a `mlir::Type`
// that is also used as the value `Type` of the attribute. Only one parameter
// of the attribute may be of this type.
class AttributeSelfTypeParameter<string desc,
                                 string derivedType = "::mlir::Type"> :
    AttrOrTypeParameter<derivedType, desc> {}

#endif // OP_BASE
