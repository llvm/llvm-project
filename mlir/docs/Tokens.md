# Tokens

[TOC]

## Overview

Intuitively, a *token* value is a pointer to an operation (via an OpResult)
or a pointer to a region (via an entry block argument). A token cannot be
forwarded: a token def-use chain cannot be obscured by ops with forwarding
semantics such as `arith.select` or `cf.br`. This allows you to always walk
back from a use and say "this token came from *that* specific op". 

A token is an SSA value that has the builtin token type. The token type is 
parameterless, opaque and carries no runtime data. A token prints as `token`.
Apart from the structural contract below, tokens are like any other SSA values.

## Design Rationale

The token type allows operations to refer to another operation without a new
parallel def-use system for operations. It reuses the existing def-use
machinery for SSA. It introduces no changes to the generic op syntax, the
bytecode infrastructure or core C++ APIs around `Operation`.

As with regular use-def chains, a token def-use chain is unidirectional. A
token use points to the token's definition and not the other way around.
Transformations can remove the use of a token without having to touch or
inspect the definition of the token.

## Structural Contract

A token use cannot be substituted with another token value: the use of a token
points directly to a specific producer. Generic transformations must not alter
or break this link. New uses of a token can be introduced safely. As a
consequence:

1. A token must not appear as a forwarded value, e.g.:
    * a forwarded result/operand of a `CallOpInterface` op,
    * an argument or result type of a `FunctionOpInterface` op (a token
      block argument *inside* a function body is fine — what is disallowed
      is forwarding tokens across the call/return boundary),
    * a successor operand or successor block argument of a
      `BranchOpInterface` op,
    * a forwarded operand to/from any region of a `RegionBranchOpInterface`
      op (iter-args, region results, yielded values), or
    * the result of any op that selects or merges values it does not
      understand (e.g. `arith.select`).

2. Given a use of a token SSA value, its definition is guaranteed to be the
   semantic producer of the token.

3. A token cannot constant-fold. No constant of token type exists.

4. Use of a token is side-effect free: a token user follows the usual `isTriviallyDead()` rules.

These properties mirror what LLVM IR already documents for its own
[`token` type](https://llvm.org/docs/LangRef.html#token-type).

## ODS Integration

Tokens are excluded from the default `AnyType` predicate, so an op that has
not opted in cannot accept a token as an arbitrary operand or result. This
restriction prevents tokens from being accidentally passed as operands with
forwarding semantics.

Three predicates are provided in `CommonTypeConstraints.td`:

| Predicate          | Accepts                              | Use when …                                                            |
| ------------------ | ------------------------------------ | ----------------------------------------------------------------------|
| `AnyType`          | any non-token type                   | the default; matches the historical meaning of "any type" pre-tokens. |
| `Token`            | only the builtin `TokenType`         | the op specifically takes a token operand/result.                     |

Example:

```tablegen
def MyConsumeOp : MyDialect_Op<"consume"> {
  let arguments = (ins Token:$scope, AnyType:$value);
}
```

## Examples

### Rejected: tokens in `AnyType` positions

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
operands likewise use `AnyType`. Because `AnyType` excludes tokens, yielding
(or returning) a token through `scf.if` (or any other op that has not
explicitly opted in) is rejected.
