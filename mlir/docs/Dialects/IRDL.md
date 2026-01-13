# 'irdl' Dialect

[TOC]

## Basics

The IRDL (*Intermediate Representation Definition Language*) dialect allows
defining MLIR dialects as MLIR programs. Nested operations are used to
represent dialect structure: dialects contain operations, types and
attributes, themselves containing type parameters, operands, results, etc.
Each of those concepts are mapped to MLIR operations in the IRDL dialect, as
shown in the example dialect below:

```mlir
irdl.dialect @cmath {
    irdl.type @complex {
        %0 = irdl.is f32
        %1 = irdl.is f64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
    }

    irdl.operation @mul {
        %0 = irdl.is f32
        %1 = irdl.is f64
        %2 = irdl.any_of(%0, %1)
        %3 = irdl.parametric @cmath::@complex<%2>
        irdl.operands(%3, %3)
        irdl.results(%3)
    }
}
```

This program defines a `cmath` dialect that defines a `complex` type, and
a `mul` operation. Both express constraints over their parameters using
SSA constraint operations. Informally, one can see those SSA values as
constraint variables that evaluate to a single type at constraint
evaluation. For example, the result of the `irdl.any_of` stored in `%2`
in the `mul` operation will collapse into either `f32` or `f64` for the
entirety of this instance of `mul` constraint evaluation. As such,
both operands and the result of `mul` must be of equal type (and not just
satisfy the same constraint). For more information, see
[constraints and combinators](#constraints-and-combinators).

In order to simplify the dialect, IRDL variables are handles over
`mlir::Attribute`. In order to support manipulating `mlir::Type`,
IRDL wraps all types in an `mlir::TypeAttr` attribute.

## Principles

The core principles of IRDL are the following, in no particular order:

- **Portability.** IRDL dialects should be self-contained, such that dialects
  can be easily distributed with minimal assumptions on which compiler 
  infrastructure (or which commit of MLIR) is used.
- **Introspection.** The IRDL dialect definition mechanism should strive
  towards offering as much introspection abilities as possible. Dialects
  should be as easy to manipulate, generate, and analyze as possible.
- **Runtime declaration support**. The specification of IRDL dialects should
  offer the ability to have them be loaded at runtime, via dynamic registration
  or JIT compilation. Compatibility with dynamic workflows should not hinder
  the ability to compile IRDL dialects into ahead-of-time declarations.
- **Reliability.** Concepts in IRDL should be consistent and predictable, with
  as much focus on high-level simplicity as possible. Consequently, IRDL
  definitions that verify should work out of the box, and those that do not
  verify should provide clear and understandable errors in all circumstances.

While IRDL simplifies IR definition, it remains an IR itself and thus does not
require to be comfortably user-writeable.

## Constraints and combinators

Attribute, type and operation verifiers are expressed in terms of constraint
variables. Constraint variables are defined as the results of constraint
operations (like `irdl.is` or constraint combinators).

Constraint variables act as variables: as such, matching against the same
constraint variable multiple times can only succeed if the matching type or 
attribute is the same as the one that previously matched. In the following
example:

```mlir
irdl.type @foo {
    %ty = irdl.any_type
    irdl.parameters(param1: %ty, param2: %ty)
}
```

only types with two equal parameters will successfully match (`foo<i32, i32>`
would match while `foo<i32, i64>` would fail, even though both i32 and i64
individually satisfy the `irdl.any_type` constraint). This constraint variable
mechanism allows to easily express a requirement on type or attribute equality.

To declare more complex verifiers, IRDL provides constraint-combinator
operations such as `irdl.any_of`, `irdl.all_of` or `irdl.parametric`. These
combinators can be used to combine constraint variables into new constraint
variables. Like all uses of constraint variables, their constraint variable
operands enforce equality of matched types of attributes as explained in the
previous paragraph.

## Motivating use cases

To illustrate the rationale behind IRDL, the following list describes examples
of intended use cases for IRDL, in no particular order:

- **Fuzzer generation.** With declarative verifier definitions, it is possible
  to compile IRDL dialects into compiler fuzzers that generate only programs
  passing verifiers.
- **Portable dialects between compiler infrastructures.** Some compiler
  infrastructures are independent from MLIR but are otherwise IR-compatible.
  Portable IRDL dialects allow to share the dialect definitions between MLIR
  and other compiler infrastructures without needing to maintain multiple
  potentially out-of-sync definitions.
- **Dialect simplification.** Because IRDL definitions can easily be 
  mechanically modified, it is possible to simplify the definition of dialects
  based on which operations are actually used, leading to smaller compilers.
- **SMT analysis.** Because IRDL dialect definitions are declarative, their
  definition can be lowered to alternative representations like SMT, allowing
  analysis of the behavior of transforms taking verifiers into account.

## Operations

[include "Dialects/IRDLOps.md"]
