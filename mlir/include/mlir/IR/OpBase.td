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

include "mlir/IR/Constraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/IR/Properties.td"
include "mlir/IR/Traits.td"
include "mlir/IR/Utils.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// OpTrait definitions
//===----------------------------------------------------------------------===//

// A trait that describes the structure of operation will be marked with
// `StructuralOpTrait` and they will be verified first.
class StructuralOpTrait;

// These classes are used to define operation specific traits.

// Specify op specific declarations and definitions in `extraOpDeclaration`
// and `extraOpDefinition` template arguments.
class NativeOpTrait<string name, list<Trait> traits = [],
                    code extraOpDeclaration = [{}],
                    code extraOpDefinition = [{}]>
    : NativeTrait<name, "Op", extraOpDeclaration, extraOpDefinition> {
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

class SingleBlockImplicitTerminatorImpl<string op>
    : ParamNativeOpTrait<"SingleBlockImplicitTerminator", op, [SingleBlock]>,
      StructuralOpTrait;

// Op's regions have a single block with the specified terminator.
class SingleBlockImplicitTerminator<string op>
    : TraitList<[SingleBlock, SingleBlockImplicitTerminatorImpl<op>]>;

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

// Uses an attribute named `operandSegmentSizes` to specify how many actual
// operand each ODS-declared operand (variadic or not) corresponds to.
// This trait is used for ops that have multiple variadic operands but do
// not know statically their size relationship. The attribute must be a 1D
// vector that has the same number of elements as the number of ODS declared
// operands. That means even if some operands are non-variadic, the attribute
// still need to have an element for its size, which is always 1.
def AttrSizedOperandSegments :
  NativeOpTrait<"AttrSizedOperandSegments">, StructuralOpTrait;
// Similar to AttrSizedOperandSegments, but used for results. The attribute
// should be named as `resultSegmentSizes`.
def AttrSizedResultSegments  :
  NativeOpTrait<"AttrSizedResultSegments">, StructuralOpTrait;

// Op attached regions have no arguments
def NoRegionArguments : NativeOpTrait<"NoRegionArguments">, StructuralOpTrait;

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
// Region definitions
//===----------------------------------------------------------------------===//

class Region<Pred condition, string descr = ""> :
    RegionConstraint<condition, descr>;

// Any region.
def AnyRegion : Region<CPred<"true">, "any region">;

// A region with the given number of blocks.
class SizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItems($_self, " # numBlocks # ")">,
  "region with " # numBlocks # " blocks"> {
  int blocks = numBlocks;
}

// A region with at least the given number of blocks.
class MinSizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItemsOrMore($_self, " # numBlocks # ")">,
  "region with at least " # numBlocks # " blocks">;

// A region with at most the given number of blocks.
class MaxSizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItemsOrLess($_self, " # numBlocks # ")">,
  "region with at most " # numBlocks # " blocks">;

// A variadic region constraint. It expands to zero or more of the base region.
class VariadicRegion<Region region>
  : Region<region.predicate, region.summary>;

//===----------------------------------------------------------------------===//
// Markers
//===----------------------------------------------------------------------===//

// Marker used to identify the region list.
def region;

// Marker used to identify the successor list.
def successor;

//===----------------------------------------------------------------------===//
// Op definitions
//===----------------------------------------------------------------------===//

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

// OpBuilder like the above, but the emitted 'build' method is marked as
// deprecated in C++. Use of it will emit a warning by the C++ compiler
// with the given reason.
class DeprecatedOpBuilder<string reason, dag p, code b = "">
  : OpBuilder<p, b>, CppDeprecated<reason>;

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

// Marker to group ops together for documentation purposes.
class OpDocGroup {
  // Single line summary of the group of ops.
  string summary;

  // Longer description of documentation group.
  string description;
}

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

  // Optional. The group of ops this op is part of.
  OpDocGroup opDocGroup = ?;

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
  // operations because those operations may be ill-formed. Use the
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

  // Whether to let ops implement their custom `readProperties` and
  // `writeProperties` methods to emit bytecode.
  bit useCustomPropertiesEncoding = 0;

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
// Common promised interface constraints
//===----------------------------------------------------------------------===//

// This constrait represents a promise or an implementation of an attr interface.
class PromisedAttrInterface<AttrInterface interface> : AttrConstraint<
  CPred<"$_self.hasPromiseOrImplementsInterface<" #
    !if(!empty(interface.cppNamespace),
        "",
        interface.cppNamespace # "::") # interface.cppInterfaceName #">()">,
  "promising or implementing the `" # interface.cppInterfaceName # "` attr interface">;

// This predicate checks if the type promises or implementats a type interface.
class HasPromiseOrImplementsTypeInterface<TypeInterface interface> :
  CPred<"$_self.hasPromiseOrImplementsInterface<" #
    !if(!empty(interface.cppNamespace),
        "",
        interface.cppNamespace # "::") # interface.cppInterfaceName #">()">;

// This constrait represents a promise or an implementation of a type interface.
class PromisedTypeInterface<TypeInterface interface> : TypeConstraint<
  HasPromiseOrImplementsTypeInterface<interface>,
  "promising or implementing the `" # interface.cppInterfaceName # "` type interface">;

//===----------------------------------------------------------------------===//
// Common op type constraints
//===----------------------------------------------------------------------===//

// These traits are for verifying properties of an op that require knowledge of
// multiple arguments or results. For verifying properties of a single argument
// or result, prefer operand type constraints.

// These traits often require including "mlir/IR/TypeUtilities.h".

// TODO: Improve the autogenerated error messages.

class Rank<string name> :
    StrFunc<"::llvm::cast<::mlir::ShapedType>($" # name # ".getType()).getRank()">;

class Shape<string name> :
    StrFunc<"::llvm::cast<::mlir::ShapedType>($" # name # ".getType()).getShape()">;

class ElementCount<string name> :
  StrFunc<"llvm::cast<::mlir::ShapedType>($" # name # ".getType())"
                                 ".getNumElements()">;

class ElementType<string name> : StrFunc<"getElementTypeOrSelf($" # name # ")">;

class AnyPred<list<string> values> :
  CPred<!if(!lt(!size(values), 1),
            "false",
            !foldl("(" # !head(values) # ")", !tail(values), acc, v,
                   acc # " || (" # v # ")"))>;

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

class AnyMatchOperatorPred<list<string> names, string operator> :
    AnyPred<!foreach(n, names, !subst("$_self", "$" # n, operator))>;

class AnyMatchOperatorTrait<list<string> names, string operator,
                            string summary> :
    PredOpTrait<
        "any of {" # !interleave(names, ", ") # "} has " # summary,
        AnyMatchOperatorPred<names, operator>> {
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

// The same as TypesMatchWith but if either `lhsArg` or `rhsArg` are optional
// and not present returns success.
class OptionalTypesMatchWith<string summary, string lhsArg, string rhsArg,
                     string transform, string comparator = "std::equal_to<>()">
  : TypesMatchWith<summary, lhsArg, rhsArg, transform,
     "!get" # snakeCaseToCamelCase<lhsArg>.ret # "()"
     # " || !get" # snakeCaseToCamelCase<rhsArg>.ret # "() || " # comparator>;

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
