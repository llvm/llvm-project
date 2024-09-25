# Operation Canonicalization

Canonicalization is an important part of compiler IR design: it makes it easier
to implement reliable compiler transformations and to reason about what is
better or worse in the code, and it forces interesting discussions about the
goals of a particular level of IR. Dan Gohman wrote
[an article](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html)
exploring these issues; it is worth reading if you're not familiar with these
concepts.

Most compilers have canonicalization passes, and sometimes they have many
different ones (e.g. instcombine, dag combine, etc in LLVM). Because MLIR is a
multi-level IR, we can provide a single canonicalization infrastructure and
reuse it across many different IRs that it represents. This document describes
the general approach, global canonicalizations performed, and provides sections
to capture IR-specific rules for reference.

[TOC]

## General Design

MLIR has a single canonicalization pass, which iteratively applies the
canonicalization patterns of all loaded dialects in a greedy way.
Canonicalization is best-effort and not guaranteed to bring the entire IR in a
canonical form. It applies patterns until either fixpoint is reached or the
maximum number of iterations/rewrites (as specified via pass options) is
exhausted. This is for efficiency reasons and to ensure that faulty patterns
cannot cause infinite looping.

Canonicalization patterns are registered with the operations themselves, which
allows each dialect to define its own set of operations and canonicalizations
together.

Some important things to think about w.r.t. canonicalization patterns:

*   The goal of canonicalization is to make subsequent analyses and
    optimizations more effective. Therefore, performance improvements are not
    necessary for canonicalization.

*   Pass pipelines should not rely on the canonicalizer pass for correctness.
    They should work correctly with all instances of the canonicalization pass
    removed.

*   Repeated applications of patterns should converge. Unstable or cyclic
    rewrites are considered a bug: they can make the canonicalizer pass less
    predictable and less effective (i.e., some patterns may not be applied) and
    prevent it from converging.

*   It is generally better to canonicalize towards operations that have fewer
    uses of a value when the operands are duplicated, because some patterns only
    match when a value has a single user. For example, it is generally good to
    canonicalize "x + x" into "x * 2", because this reduces the number of uses
    of x by one.

*   It is always good to eliminate operations entirely when possible, e.g. by
    folding known identities (like "x + 0 = x").

*   Pattens with expensive running time (i.e. have O(n) complexity) or
    complicated cost models don't belong to canonicalization: since the
    algorithm is executed iteratively until fixed-point we want patterns that
    execute quickly (in particular their matching phase).

*   Canonicalize shouldn't lose the semantic of original operation: the original
    information should always be recoverable from the transformed IR.

For example, a pattern that transform

```
  %transpose = linalg.transpose
      ins(%input : tensor<1x2x3xf32>)
      outs(%init1 : tensor<2x1x3xf32>)
      dimensions = [1, 0, 2]
  %out = linalg.transpose
      ins(%tranpose: tensor<2x1x3xf32>)
      outs(%init2 : tensor<3x1x2xf32>)
      permutation = [2, 1, 0]
```

to

```
  %out= linalg.transpose
      ins(%input : tensor<1x2x3xf32>)
      outs(%init2: tensor<3x1x2xf32>)
      permutation = [2, 0, 1]
```

is a good canonicalization pattern because it removes a redundant operation,
making other analysis optimizations and more efficient.

## Globally Applied Rules

These transformations are applied to all levels of IR:

*   Elimination of operations that have no side effects and have no uses.

*   Constant folding - e.g. "(addi 1, 2)" to "3". Constant folding hooks are
    specified by operations.

*   Move constant operands to commutative operators to the right side - e.g.
    "(addi 4, x)" to "(addi x, 4)".

*   `constant-like` operations are uniqued and hoisted into the entry block of
    the first parent barrier region. This is a region that is either isolated
    from above, e.g. the entry block of a function, or one marked as a barrier
    via the `shouldMaterializeInto` method on the `DialectFoldInterface`.

## Defining Canonicalizations

Two mechanisms are available with which to define canonicalizations;
general `RewritePattern`s and the `fold` method.

### Canonicalizing with `RewritePattern`s

This mechanism allows for providing canonicalizations as a set of
`RewritePattern`s, either imperatively defined in C++ or declaratively as
[Declarative Rewrite Rules](DeclarativeRewrites.md). The pattern rewrite
infrastructure allows for expressing many different types of canonicalizations.
These transformations may be as simple as replacing a multiplication with a
shift, or even replacing a conditional branch with an unconditional one.

In [ODS](DefiningDialects/Operations.md), an operation can set the `hasCanonicalizer` bit or
the `hasCanonicalizeMethod` bit to generate a declaration for the
`getCanonicalizationPatterns` method:

```tablegen
def MyOp : ... {
  // I want to define a fully general set of patterns for this op.
  let hasCanonicalizer = 1;
}

def OtherOp : ... {
  // A single "matchAndRewrite" style RewritePattern implemented as a method
  // is good enough for me.
  let hasCanonicalizeMethod = 1;
}
```

Canonicalization patterns can then be provided in the source file:

```c++
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<...>(...);
}

LogicalResult OtherOp::canonicalize(OtherOp op, PatternRewriter &rewriter) {
  // patterns and rewrites go here.
  return failure();
}
```

See the [quickstart guide](Tutorials/QuickstartRewrites.md) for information on
defining operation rewrites.

### Canonicalizing with the `fold` method

The `fold` mechanism is an intentionally limited, but powerful mechanism that
allows for applying canonicalizations in many places throughout the compiler.
For example, outside of the canonicalizer pass, `fold` is used within the
[dialect conversion infrastructure](DialectConversion.md) as a legalization
mechanism, and can be invoked directly anywhere with an `OpBuilder` via
`OpBuilder::createOrFold`.

`fold` has the restriction that no new operations may be created, and only the
root operation may be replaced (but not erased). It allows for updating an
operation in-place, or returning a set of pre-existing values (or attributes) to
replace the operation with. This ensures that the `fold` method is a truly
"local" transformation, and can be invoked without the need for a pattern
rewriter.

In [ODS](DefiningDialects/Operations.md), an operation can set the `hasFolder` bit to generate
a declaration for the `fold` method. This method takes on a different form,
depending on the structure of the operation.

```tablegen
def MyOp : ... {
  let hasFolder = 1;
}
```

If the operation has a single result the following will be generated:

```c++
/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return nullptr.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return the operation itself.
///  3. They can return an existing value or attribute that can be used instead
///     of the operation. The caller will remove the operation and use that
///     result instead.
///
OpFoldResult MyOp::fold(FoldAdaptor adaptor) {
  ...
}
```

Otherwise, the following is generated:

```c++
/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return failure.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return success.
///  3. They can return a list of existing values or attribute that can be used
///     instead of the operation. In this case, fill in the results list and
///     return success. The results list must correspond 1-1 with the results of
///     the operation, partial folding is not supported. The caller will remove
///     the operation and use those results instead.
///
/// Note that this mechanism cannot be used to remove 0-result operations.
LogicalResult MyOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results) {
  ...
}
```

In the above, for each method a `FoldAdaptor` is provided with getters for
each of the operands, returning the corresponding constant attribute. These
operands are those that implement the `ConstantLike` trait. If any of the
operands are non-constant, a null `Attribute` value is provided instead. For
example, if MyOp provides three operands [`a`, `b`, `c`], but only `b` is
constant then `adaptor` will return Attribute() for `getA()` and `getC()`,
and b-value for `getB()`.

Also above, is the use of `OpFoldResult`. This class represents the possible
result of folding an operation result: either an SSA `Value`, or an
`Attribute`(for a constant result). If an SSA `Value` is provided, it *must*
correspond to an existing value. The `fold` methods are not permitted to
generate new `Value`s. There are no specific restrictions on the form of the
`Attribute` value returned, but it is important to ensure that the `Attribute`
representation of a specific `Type` is consistent.

When the `fold` hook on an operation is not successful, the dialect can
provide a fallback by implementing the `DialectFoldInterface` and overriding
the fold hook.

#### Generating Constants from Attributes

When a `fold` method returns an `Attribute` as the result, it signifies that
this result is "constant". The `Attribute` is the constant representation of the
value. Users of the `fold` method, such as the canonicalizer pass, will take
these `Attribute`s and materialize constant operations in the IR to represent
them. To enable this materialization, the dialect of the operation must
implement the `materializeConstant` hook. This hook takes in an `Attribute`
value, generally returned by `fold`, and produces a "constant-like" operation
that materializes that value.

In [ODS](DefiningDialects/_index.md), a dialect can set the `hasConstantMaterializer` bit
to generate a declaration for the `materializeConstant` method.

```tablegen
def MyDialect : ... {
  let hasConstantMaterializer = 1;
}
```

Constants can then be materialized in the source file:

```c++
/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the value generated to represent the constant value.
/// Otherwise, it should return nullptr on failure.
Operation *MyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  ...
}
```
