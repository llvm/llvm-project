//===-- Constraints.td - Constraints definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines constraints/predicates for verifiers.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTRAINTS
#define CONSTRAINTS

include "mlir/IR/Utils.td"


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
//   `CPred<"::llvm::isa<BoolAttr>($_self)">` (see the following sections for details).
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
                     string cppTypeParam = "::mlir::Type",
                     string cppFunctionNameParam = ""> :
    Constraint<predicate, summary> {
  // The name of the C++ Type class if known, or Type if not.
  string cppType = cppTypeParam;
  // The name of the C++ function that is generated for this type constraint.
  // If empty, no C++ function is generated.
  string cppFunctionName = cppFunctionNameParam;
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

#endif // CONSTRAINTS
