# Tokens

[TOC]

## Overview

Intuitively, a *token* value is a pointer to an operation (via an OpResult)
or a pointer to a region (via an entry block argument). A token cannot be
forwarded: a token def-use chain cannot be obscured by ops with forwarding
semantics such as `arith.select` or `cf.br`. This allows you to always walk
back from a use and say "this token came from *that* specific op". The exact
structural contract is specified in the
[LangRef section on tokens](LangRef.md#token-type).

A token is an SSA value that has the builtin token type. The token type is 
parameterless, opaque and carries no runtime data. Apart from the structural
contract specified in the LangRef, tokens are like any other SSA values.

## Design Rationale

The token type allows operations to refer to another operation without a new
parallel def-use system for operations. It reuses the existing def-use
machinery for SSA. It introduces no changes to the generic op syntax, the
bytecode infrastructure or core C++ APIs around `Operation`.

As with regular def-use chains, a token def-use chain is unidirectional. A
token use points to the token's definition and not the other way around.
Transformations can remove the use of a token without having to touch or
inspect the definition of the token.

Because tokens are SSA values, they cannot cross `IsolatedFromAbove` region
boundaries. This is intentional: it allows passes to process isolated regions
concurrently without racing on def-use chains. When a token-like dependency
must cross such a boundary, another mechanism must be used (e.g. a symbolic
reference using an attribute).

## ODS Integration

Tokens are excluded from the default `AnyType` predicate, so an op that has
not opted in cannot accept a token as an arbitrary operand or result. This
restriction prevents tokens from being accidentally passed as operands with
forwarding semantics.

Two predicates are provided in `CommonTypeConstraints.td`:

| Predicate          | Accepts                              | Use when …                                                            |
| ------------------ | ------------------------------------ | ----------------------------------------------------------------------|
| `AnyType`          | any non-token type                   | the default; matches the historical meaning of "any type" pre-tokens. |
| `Token`            | only the builtin `TokenType`         | the op specifically takes a token operand/result.                     |

Example:

```tablegen
def MyProduceOp : MyDialect_Op<"produce"> {
  let results = (outs Token:$token);
}

def MyConsumeOp : MyDialect_Op<"consume"> {
  let arguments = (ins Token:$scope, AnyType:$value);
}
```

ODS automatically adds `TokenProducerTrait` when an op declares a `Token`-typed
result, and `TokenConsumerTrait` when it declares a `Token`-typed operand. The
traits must be listed manually when tokens appear as entry block arguments and
for ops that are not defined in ODS.

Region entry block arguments of `token` type are also token producers and
require the parent operation to define `TokenProducerTrait`. Token block
arguments in non-entry blocks are rejected.

## Examples

### Non-forwarding Semantics

The [LangRef](LangRef.md#token-type) requires that a token never appears as a
forwarded value. For example, you cannot use a token like this:

* a forwarded result or operand of a `CallOpInterface` op;
* an argument or result type of a `FunctionOpInterface` op;
* a successor operand of a `BranchOpInterface` op;
* a block argument of a non-entry block;
* a forwarded operand to or from any region of a `RegionBranchOpInterface`
  op (iter-args, region results, or yielded values); or
* the result of any op that selects or merges values it does not understand
  (e.g. `arith.select`).

### ODS-based Verification: Tokens Rejected in `AnyType` Positions

`scf.yield` operands have forwarding semantics. A token cannot be yielded from
a branch or a loop.

```mlir
// error: 'scf.if' op result #0 must be variadic of any non-token type,
//        but got 'token'
%t = scf.if %cond -> token {
  %a = my.token.produce : token
  scf.yield %a : token
} else {
  %b = my.token.produce : token
  scf.yield %b : token
}
```

`scf.if`'s results are declared with `Variadic<AnyType>` and `scf.yield`'s
operands likewise use `AnyType`. Because `AnyType` excludes tokens, both
`scf.if` and `scf.yield` fail verification.
