# PDLL - PDL Language

This document details the PDL Language (PDLL), a custom frontend language for
writing pattern rewrites targeting MLIR.

Note: This document assumes a familiarity with MLIR concepts; more specifically
the concepts detailed within the
[MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) and
[Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/OpDefinitions/)
documentation.

[TOC]

## Introduction

Pattern matching is an extremely important component within MLIR, as it
encompasses many different facets of the compiler. From canonicalization, to
optimization, to conversion; every MLIR based compiler will heavily rely on the
pattern matching infrastructure in some capacity.

The PDL Language (PDLL) provides a declarative pattern language designed from
the ground up for representing MLIR pattern rewrites. PDLL is designed to
natively support writing matchers on all of MLIRs constructs via an intuitive
interface that may be used for both ahead-of-time (AOT) and just-in-time (JIT)
pattern compilation.

## Rationale

This section provides details on various design decisions, their rationale, and
alternatives considered when designing PDLL. Given the nature of software
development, this section may include references to areas of the MLIR compiler
that no longer exist.

### Why build a new language instead of improving TableGen DRR?

Note: This section assumes familiarity with
[TDRR](https://mlir.llvm.org/docs/DeclarativeRewrites/), please refer the
relevant documentation before continuing.

Tablegen DRR (TDRR), i.e.
[Table-driven Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/),
is a declarative DSL for defining MLIR pattern rewrites within the
[TableGen](https://llvm.org/docs/TableGen/index.html) language. This
infrastructure is currently the main way in which patterns may be defined
declaratively within MLIR. TDRR utilizes TableGen's `dag` support to enable
defining MLIR patterns that fit nicely within a DAG structure; in a similar way
in which tablegen has been used to defined patterns for LLVM's backend
infrastructure (SelectionDAG/Global Isel/etc.). Unfortunately however, the
TableGen language is not as amenable to the structure of MLIR patterns as it has
been for LLVM.

The issues with TDRR largely stem from the use of TableGen as the host language
for the DSL. These issues have risen from a mismatch in the structure of
TableGen compared to the structure of MLIR, and from TableGen having different
motivational goals than MLIR. A majority (or all depending on how stubborn you
are) of the issues that we've come across with TDRR have been addressable in
some form; the sticking point here is that the solutions to these problems have
often been more "creative" than we'd like. This is a problem, and why we decided
not to invest a larger effort into improving TDRR; users generally don't want
"creative" APIs, they want something that is intuitive to read/write.

To highlight some of these issues, below we will take a tour through some of the
problems that have arisen, and how we "fixed" them.

#### Multi-result operations

MLIR natively supports a variable number of operation results. For the DAG based
structure of TDRR, any form of multiple results (operations in this instance)
creates a problem. This is because the DAG wants a single root node, and does
not have nice facilities for indexing or naming the multiple results. Let's take
a look at a quick example to see how this manifests:

```tablegen
// Suppose we have a three result operation, defined as seen below.
def ThreeResultOp : Op<"three_result_op"> {
    let arguments = (ins ...);

    let results = (outs
      AnyTensor:$output1,
      AnyTensor:$output2,
      AnyTensor:$output3
    );
}

// To bind the results of `ThreeResultOp` in a TDRR pattern, we bind all results
// to a single name and use a special naming convention: `__N`, where `N` is the
// N-th result.
def : Pattern<(ThreeResultOp:$results ...),
              [(... $results__0), ..., (... $results__2), ...]>;
```

In TDRR, we "solved" the problem of accessing multiple results, but this isn't a
very intuitive interface for users. Magical naming conventions obfuscate the
code and can easily introduce bugs and other errors. There are various things
that we could try to improve this situation, but there is a fundamental limit to
what we can do given the limits of the TableGen dag structure. In PDLL, however,
we have the freedom and flexibility to provide a proper interface into
operations, regardless of their structure:

```pdll
// Import our definition of `ThreeResultOp`.
#include "ops.td"

Pattern {
  ...

  // In PDLL, we can directly reference the results of an operation variable.
  // This provides a closer mental model to what the user expects.
  let threeResultOp = op<my_dialect.three_result_op>;
  let userOp = op<my_dialect.user_op>(threeResultOp.output1, ..., threeResultOp.output3);

  ...
}
```

#### Constraints

In TDRR, the match dag defines the general structure of the input IR to match.
Any non-structural/non-type constraints on the input are generally relegated to
a list of constraints specified after the rewrite dag. For very simple patterns
this may suffice, but with larger patterns it becomes quite problematic as it
separates the constraint from the entity it constrains and negatively impacts
the readability of the pattern. As an example, let's look at a simple pattern
that adds additional constraints to its inputs:

```tablegen
// Suppose we have a two result operation, defined as seen below.
def TwoResultOp : Op<"two_result_op"> {
    let arguments = (ins ...);

    let results = (outs
      AnyTensor:$output1,
      AnyTensor:$output2
    );
}

// A simple constraint to check if a value is use_empty.
def HasNoUseOf: Constraint<CPred<"$_self.use_empty()">, "has no use">;

// Check if two values have a ShapedType with the same element type.
def HasSameElementType : Constraint<
    CPred<"$0.getType().cast<ShapedType>().getElementType() == "
          "$1.getType().cast<ShapedType>().getElementType()">,
    "values have same element type">;

def : Pattern<(TwoResultOp:$results $input),
              [(...), (...)],
              [(HasNoUseOf:$results__1),
               (HasSameElementType $results__0, $input)]>;
```

Above, when observing the constraints we need to search through the input dag
for the inputs (also keeping in mind the magic naming convention for multiple
results). For this simple pattern it may be just a few lines above, but complex
patterns often grow to 10s of lines long. In PDLL, these constraints can be
applied directly on or next to the entities they apply to:

```pdll
// The same constraints that we defined above:
Constraint HasNoUseOf(value: Value) [{
  return success(value.use_empty());
}];
Constraint HasSameElementType(value1: Value, value2: Value) [{
  return success(value1.getType().cast<ShapedType>().getElementType() ==
                 value2.getType().cast<ShapedType>().getElementType());
}];

Pattern {
  // In PDLL, we can apply the constraint as early (or as late) as we want. This
  // enables better structuring of the matcher code, and improves the
  // readability/maintainability of the pattern.
  let op = op<my_dialect.two_result_op>(input: Value);
  HasNoUseOf(op.output2);
  HasSameElementType(input, op.output2);

  // ...
}
```

#### Replacing Multiple Operations

Often times a pattern will transform N number of input operations into N number
of result operations. In PDLL, replacing multiple operations is as simple as
adding two [`replace` statements](#replace-statement). In TDRR, the situation is
a bit more nuanced. Given the single root structure of the TableGen dag,
replacing a non-root operation is not nicely supported. It currently isn't
natively possible, and instead requires using multiple patterns. We could
potentially add another special rewrite directive, or extend `replaceWithValue`,
but this simply highlights how even a basic IR transformation is muddled by the
complexity of the host language.

### Why not build a DSL in "X"?

Yes! Well yes and no. To understand why, we have to consider what types of users
we are trying to serve and what constraints we enforce upon them. The goal of
PDLL is to provide a default and effective pattern language for MLIR that all
users of MLIR can interact with immediately, regardless of their host
environment. This language is available with no extra dependencies and comes
"free" along with MLIR. If we were to use an existing host language to build our
new DSL, we would need to make compromises along with it depending on the
language. For some, there are questions of how to enforce matching environments
(python2 or python3?, which version?), performance considerations, integration,
etc. As an LLVM project, this could also mean enforcing a new language
dependency on the users of MLIR (many of which may not want/need such a
dependency otherwise). Another issue that comes along with any DSL that is
embeded in another language: mitigating the user impedance mismatch between what
the user expects from the host language and what our "backend" supports. For
example, the PDL IR abstraction only contains limited support for control flow.
If we were to build a DSL in python, we would need to ensure that complex
control flow is either handled completely or effectively errors out. Even with
ideal error handling, not having the expected features available creates user
frustration. In addition to the environment constraints, there is also the issue
of language tooling. With PDLL we intend to build a very robust and modern
toolset that is designed to cater the needs of pattern developers, including
code completion, signature help, and many more features that are specific to the
problem we are solving. Integrating custom language tooling into existing
languages can be difficult, and in some cases impossible (as our DSL would
merely be a small subset of the existing language).

These various points have led us to the initial conclusion that the most
effective tool we can provide for our users is a custom tool designed for the
problem at hand. With all of that being said, we understand that not all users
have the same constraints that we have placed upon ourselves. We absolutely
encourage and support the existence of various PDL frontends defined in
different languages. This is one of the original motivating factors around
building the PDL IR abstraction in the first place; to enable innovation and
flexibility for our users (and in turn their users). For some, such as those in
research and the Machine Learning space, they may already have a certain
language (such as Python) heavily integrated into their workflow. For these
users, a PDL DSL in their language may be ideal and we will remain committed to
supporting and endorsing that from an infrastructure point-of-view.

## Language Specification

Note: PDLL is still under active development, and the designs discussed below
are not necessarily final and may be subject to change.

The design of PDLL is heavily influenced and centered around the
[PDL IR abstraction](https://mlir.llvm.org/docs/Dialects/PDLOps/), which in turn
is designed as an abstract model of the core MLIR structures. This leads to a
design and structure that feels very similar to if you were directly writing the
IR you want to match.

### Includes

PDLL supports an `include` directive to import content defined within other
source files. There are two types of files that may be included: `.pdll` and
`.td` files.

#### `.pdll` includes

When including a `.pdll` file, the contents of that file are copied directly into
the current file being processed. This means that any patterns, constraints,
rewrites, etc., defined within that file are processed along with those within
the current file.

#### `.td` includes

When including a `.td` file, PDLL will automatically import any pertinent
[ODS](https://mlir.llvm.org/docs/OpDefinitions/) information within that file.
This includes any defined operations, constraints, interfaces, and more, making
them implicitly accessible within PDLL. This is important, as ODS information
allows for certain PDLL constructs, such as the
[`operation` expression](#operation), to become much more powerful.

### Patterns

In any pattern descriptor language, pattern definition is at the core. In PDLL,
patterns start with `Pattern` optionally followed by a name and a set of pattern
metadata, and finally terminated by a pattern body. A few simple examples are
shown below:

```pdll
// Here we have defined an anonymous pattern:
Pattern {
  // Pattern bodies are separated into two components:
  // * Match Section
  //    - Describes the input IR.
  let root = op<toy.reshape>(op<toy.reshape>(arg: Value));
  
  // * Rewrite Section
  //    - Describes how to transform the IR.
  //    - Last statement starts the rewrite.
  replace root with op<toy.reshape>(arg);
}

// Here we have defined a pattern named `ReshapeReshapeOptPattern` with a
// benefit of 10:
Pattern ReshapeReshapeOptPattern with benefit(10) {
  replace op<toy.reshape>(op<toy.reshape>(arg: Value))
    with op<toy.reshape>(arg);
}
```

After the definition of the pattern metadata, we specify the pattern body. The
structure of a pattern body is comprised of two main sections, the `match`
section and the `rewrite` section. The `match` section of a pattern describes
the expected input IR, whereas the `rewrite` section describes how to transform
that IR. This distinction is an important one to make, as PDLL handles certain
variables and expressions differently within the different sections. When
relevant in each of the sections below, we shall explicitly call out any
behavioral differences.

The general layout of the `match` and `rewrite` section is as follows: the
*last* statement of the pattern body is required to be a
[`operation rewrite statement`](#operation-rewrite-statements), and denotes the
`rewrite` section; every statement before denotes the `match` section.

#### Pattern metadata

Rewrite patterns in MLIR have a set of metadata that allow for controlling
certain behaviors, and providing information to the rewrite driver applying the
pattern. In PDLL, a pattern can provide a non-default value for this metadata
after the pattern name. Below, examples are shown for the different types of
metadata supported:

##### Benefit

The benefit of a Pattern is an integer value that represents the "benefit" of
matching that pattern. It is used by pattern drivers to determine the relative
priorities of patterns during application; a pattern with a higher benefit is
generally applied before one with a lower benefit.

In PDLL, a pattern has a default benefit set to the number of input operations,
i.e. the number of distinct `Op` expressions/variables, in the match section. This
rule is driven by an observation that larger matches are more beneficial than smaller
ones, and if a smaller one is applied first the larger one may not apply anymore.
Patterns can override this behavior by specifying the benefit in the metadata section
of the pattern:

```pdll
// Here we specify that this pattern has a benefit of `10`, overriding the
// default behavior.
Pattern with benefit(10) {
  ...
}
```

##### Bounded Rewrite Recursion

During pattern application, there are situations in which a pattern may be
applicable to the result of a previous application of that same pattern. If the
pattern does not properly handle this recusive application, the pattern driver
could become stuck in an infinite loop of application. To prevent this, patterns
by-default are assumed to not have proper recursive bounding and will not be
recursively applied. A pattern can signal that it does have proper handling for
recursion by specifying the `recusion` flag in the pattern metadata section:

```pdll
// Here we signal that this pattern properly bounds recursive application.
Pattern with recusion {
  ...
}
```

#### Single Line "Lambda" Body

Patterns generally define their body using a compound block of statements, as
shown below:

```pdll
Pattern {
  replace op<my_dialect.foo>(operands: ValueRange) with operands;
}
```

Patterns also support a lambda-like syntax for specifying simple single line
bodies. The lambda body of a Pattern expects a single
[operation rewrite statement](#operation-rewrite-statements):

```pdll
Pattern => replace op<my_dialect.foo>(operands: ValueRange) with operands;
```

### Variables

Variables in PDLL represent specific instances of IR entities, such as `Value`s,
`Operation`s, `Type`s, etc. Consider the simple pattern below:

```pdll
Pattern {
  let value: Value;
  let root = op<mydialect.foo>(value);

  replace root with value;
}
```

In this pattern we define two variables, `value` and `root`, using the `let`
statement. The `let` statement allows for defining variables and constraining
them. Every variable in PDLL is of a certain type, which defines the type of IR
entity the variable represents. The type of a variable may be determined via
either a constraint, or an initializer expression.

#### Variable "Binding"

In addition to having a type, variables must also be "bound", either via an initializer
expression or to a non-native constraint or rewrite use within the `match` section of the
pattern. "Binding" a variable contextually identifies that variable within either the
input (i.e. `match` section) or output (i.e. `rewrite` section) IR. In the `match` section,
this allows for building the match tree from the pattern's root operation, which must be
"bound" to the [operation rewrite statement](#operation-rewrite-statements) that denotes the
`rewrite` section of the pattern. All non-root variables within the `match`
section must be bound in some way to the "root" operation. To help illustrate
the concept, let's take a look at a quick example. Consider the `.mlir` snippet
below:

```mlir
func.func @baz(%arg: i32) {
  %result = my_dialect.foo %arg, %arg -> i32
}
```

Say that we want to write a pattern that matches `my_dialect.foo` and replaces
it with its unique input argument. A naive way to write this pattern in PDLL is
shown below:

```pdll
Pattern {
  // ** match section ** //
  let arg: Value;
  let root = op<my_dialect.foo>(arg, arg);

  // ** rewrite section ** //
  replace root with arg;
}
```

In the above pattern, the `arg` variable is "bound" to the first and second operands
of the `root` operation. Every use of `arg` is constrained to be the same `Value`, i.e.
the first and second operands of `root` will be constrained to refer to the same input
Value. The same is true for the `root` operation, it is bound to the "root" operation of the
pattern as it is used in input of the top-level [`replace` statement](#replace-statement)
of the `rewrite` section of the pattern. Writing this pattern using the C++ API, the concept
of "binding" becomes more clear:

```c++
struct Pattern : public OpRewritePattern<my_dialect::FooOp> {
  LogicalResult matchAndRewrite(my_dialect::FooOp root, PatternRewriter &rewriter) {
    Value arg = root->getOperand(0);
    if (arg != root->getOperand(1))
      return failure();

    rewriter.replaceOp(root, arg);
    return success();
  }
};
```

If a variable is not "bound" properly, PDLL won't be able to identify what value
it would correspond to in the IR. As a final example, let's consider a variable
that hasn't been bound:

```pdll
Pattern {
  // ** match section ** //
  let arg: Value;
  let root = op<my_dialect.foo>

  // ** rewrite section ** //
  replace root with arg;
}
```

If we were to write this exact pattern in C++, we would end up with:

```c++
struct Pattern : public OpRewritePattern<my_dialect::FooOp> {
  LogicalResult matchAndRewrite(my_dialect::FooOp root, PatternRewriter &rewriter) {
    // `arg` was never bound, so we don't know what input Value it was meant to
    // correspond to.
    Value arg;

    rewriter.replaceOp(root, arg);
    return success();
  }
};
```

#### Variable Constraints

```pdll
// This statement defines a variable `value` that is constrained to be a `Value`.
let value: Value;

// This statement defines a variable `value` that is constrained to be a `Value`
// *and* constrained to have a single use.
let value: [Value, HasOneUse];
```

Any number of single entity constraints may be attached directly to a variable
upon declaration. Within the `matcher` section, these constraints may add
additional checks on the input IR. Within the `rewriter` section, constraints
are *only* used to define the type of the variable. There are a number of
builtin constraints that correlate to the core MLIR constructs: `Attr`, `Op`,
`Type`, `TypeRange`, `Value`, `ValueRange`. Along with these, users may define
custom constraints that are implemented within PDLL, or natively (i.e. outside
of PDLL). See the general [Constraints](#constraints) section for more detailed
information.

#### Inline Variable Definition

Along with the `let` statement, variables may also be defined inline by
specifying the constraint list along with the desired variable name in the first
place that the variable would be used. After definition, the variable is visible
from all points forward. See below for an example:

```pdll
// `value` is used as an operand to the operation `root`:
let value: Value;
let root = op<my_dialect.foo>(value);
replace root with value;

// `value` could also be defined "inline":
let root = op<my_dialect.foo>(value: Value);
replace root with value;
```

Note that the point of definition of an inline variable is the point of reference,
meaning that an inline variable can be used immediately in the same parent
expression within which it was defined:

```pdll
let root = op<my_dialect.foo>(value: Value, _: Value, value);
replace root with value;
```

##### Wildcard Variable Definition

Often times when defining a variable inline, the variable isn't intended to be
used anywhere else in the pattern. For example, this may happen if you want to
attach constraints to a variable but have no other use for it. In these
situations, the "wildcard" variable can be used to remove the need to provide a
name, as "wildcard" variables are not visible outside of the point of
definition. An example is shown below:

```pdll
Pattern {
  let root = op<my_dialect.foo>(arg: Value, _: Value, _: [Value, I64Value], arg);
  replace root with arg;
}
```

In the above example, the second operand isn't needed for the pattern but we
need to provide it to signal that a second operand does exist (we just don't
care what it is in this pattern).

### Operation Expression

An operation expression in PDLL represents an MLIR operation. In the `match`
section of the pattern, this expression models one of the input operations to
the pattern. In the `rewrite` section of the pattern, this expression models one
of the operations to create. The general structure of the operation expression
is very similar to that of the "generic form" of textual MLIR assembly:

```pdll
let root = op<my_dialect.foo>(operands: ValueRange) {attr = attr: Attr} -> (resultTypes: TypeRange);
```

Let's walk through each of the different components of the expression:

#### Operation name

The operation name signifies which type of MLIR Op this operation corresponds
to. In the `match` section of the pattern, the name may be elided. This would
cause this pattern to match *any* operation type that satifies the rest of the
constraints of the operation. In the `rewrite` section, the name is required.

```pdll
// `root` corresponds to an instance of a `my_dialect.foo` operation.
let root = op<my_dialect.foo>;

// `root` could be an instance of any operation type.
let root = op<>;
```

#### Operands

The operands section corresponds to the operands of the operation. This section
of an operation expression may be elided, which within a `match` section means
that the operands are not constrained in any way. If elided within a `rewrite`
section, the operation is treated as having no operands. When present, the
operands of an operation expression are interpreted in the following ways:

1) A single instance of type `ValueRange`:

In this case, the single range is treated as all of the operands of the
operation:

```pdll
// Define an instance with single range of operands.
let root = op<my_dialect.foo>(allOperands: ValueRange);
```

2) A variadic number of either `Value` or `ValueRange`:

In this case, the inputs are expected to correspond with the operand groups as
defined on the operation in ODS.

Given the following operation definition in ODS:

```tablegen
def MyIndirectCallOp {
  let arguments = (ins FunctionType:$call, Variadic<AnyType>:$args);
}
```

We can match the operands as so:

```pdll
let root = op<my_dialect.indirect_call>(call: Value, args: ValueRange);
```

#### Results

The results section corresponds to the result types of the operation. This section
of an operation expression may be elided, which within a `match` section means
that the result types are not constrained in any way. If elided within a `rewrite`
section, the results of the operation are [inferred](#inferred-results). When present,
the result types of an operation expression are interpreted in the following ways:

1) A single instance of type `TypeRange`:

In this case, the single range is treated as all of the result types of the
operation:

```pdll
// Define an instance with single range of types.
let root = op<my_dialect.foo> -> (allResultTypes: TypeRange);
```

2) A variadic number of either `Type` or `TypeRange`:

In this case, the inputs are expected to correspond with the result groups as
defined on the operation in ODS.

Given the following operation definition in ODS:

```tablegen
def MyOp {
  let results = (outs SomeType:$result, Variadic<SomeType>:$otherResults);
}
```

We can match the result types as so:

```pdll
let root = op<my_dialect.op> -> (result: Type, otherResults: TypeRange);
```

#### Inferred Results

Within the `rewrite` section of a pattern, the result types of an
operation are inferred if they are elided or otherwise not
previously bound. The ["variable binding"](#variable-binding) section above
discusses the concept of "binding" in more detail. Below are various examples
that build upon this to help showcase how a result type may be "bound":

* Binding to a [constant](#type-expression):

```pdll
op<my_dialect.op> -> (type<"i32">);
```

* Binding to types within the `match` section:

```pdll
Pattern {
  replace op<dialect.inputOp> -> (resultTypes: TypeRange)
    with op<dialect.outputOp> -> (resultTypes);
}
```

* Binding to previously inferred types:

```pdll
Pattern {
  rewrite root: Op with {
    // `resultTypes` here is *not* yet bound, and will be inferred when
    // creating `dialect.op`. Any uses of `resultTypes` after this expression,
    // will use the types inferred when creating this operation.
    op<dialect.op> -> (resultTypes: TypeRange);

    // `resultTypes` here is bound to the types inferred when creating `dialect.op`.
    op<dialect.bar> -> (resultTypes);
  };
}
```

* Binding to a [`Native Rewrite`](#native-rewriters) method result:

```pdll
Rewrite BuildTypes() -> TypeRange;

Pattern {
  rewrite root: Op with {
    op<dialect.op> -> (BuildTypes());
  };
}
```

Below are the set of contexts in which result type inferrence is supported:

##### Inferred Results of Replacement Operation

Replacements have the invariant that the types of the replacement values must
match the result types of the input operation. This means that when replacing
one operation with another, the result types of the replacement operation may
be inferred from the result types of the operation being replaced. For example,
consider the following pattern:

```pdll
Pattern => replace op<dialect.inputOp> with op<dialect.outputOp>;
```

This pattern could be written in a more explicit way as:

```pdll
Pattern {
  replace op<dialect.inputOp> -> (resultTypes: TypeRange)
    with op<dialect.outputOp> -> (resultTypes);
}
```

##### Inferred Results with InferTypeOpInterface

`InferTypeOpInterface` is an interface that enables operations to infer its result
types from its input attributes, operands, regions, etc. When the result types of
an operation cannot be inferred from any other context, this interface is invoked
to infer the result types of the operation.

#### Attributes

The attributes section of the operation expression corresponds to the attribute
dictionary of the operation. This section of an operation expression may be
elided, in which case the attributes are not constrained in any way. The
composition of this component maps exactly to how attribute dictionaries are
structured in the MLIR textual assembly format:

```pdll
let root = op<my_dialect.foo> {attr1 = attrValue: Attr, attr2 = attrValue2: Attr};
```

Within the `{}` attribute entries are specified by an identifier or string name,
corresponding to the attribute name, followed by an assignment to the attribute
value. If the attribute value is elided, the value of the attribute is
implicitly defined as a
[`UnitAttr`](https://mlir.llvm.org/docs/Dialects/Builtin/#unitattr).

```pdll
let unitConstant = op<my_dialect.constant> {value};
```

##### Accessing Operation Results

In multi-operation patterns, the result of one operation often feeds as an input
into another. The result groups of an operation may be accessed by name or by
index via the `.` operator:

Note: Remember to import the definition of your operation via
[include](#`.td`_includes) to ensure it is visible to PDLL.

Given the following operation definition in ODS:

```tablegen
def MyResultOp {
  let results = (outs SomeType:$result);
}
def MyInputOp {
  let arguments = (ins SomeType:$input, SomeType:$input);
}
```

We can write a pattern where `MyResultOp` feeds into `MyInputOp` as so:

```pdll
// In this example, we use both `result`(the name) and `0`(the index) to refer to
// the first result group of `resultOp`.
// Note: If we elide the result types section within the match section, it means
//       they aren't constrained, not that the operation has no results.
let resultOp = op<my_dialect.result_op>;
let inputOp = op<my_dialect.input_op>(resultOp.result, resultOp.0);
```

Along with result name access, variables of `Op` type may implicitly convert to
`Value` or `ValueRange`. If these variables are registered (has ODS entry), they
are converted to `Value` when they are known to only have one result, otherwise
they will be converted to `ValueRange`:

```pdll
// `resultOp` may also convert implicitly to a Value for use in `inputOp`:
let resultOp = op<my_dialect.result_op>;
let inputOp = op<my_dialect.input_op>(resultOp);

// We could also inline `resultOp` directly:
let inputOp = op<my_dialect.input_op>(op<my_dialect.result_op>);
```

#### Unregistered Operations

A variable of unregistered op is still available for numeric result indexing.
Given that we don't have knowledge of its result groups, numeric indexing
returns a Value corresponding to the individual result at the given index.

```pdll
// Use the index `0` to refer to the first result value of the unregistered op.
let inputOp = op<my_dialect.input_op>(op<my_dialect.unregistered_op>.0);
```

### Attribute Expression

An attribute expression represents a literal MLIR attribute. It allows for
statically specifying an MLIR attribute to use, by specifying the textual form
of that attribute.

```pdll
let trueConstant = op<arith.constant> {value = attr<"true">};

let applyResult = op<affine.apply>(args: ValueRange) {map = attr<"affine_map<(d0, d1) -> (d1 - 3)>">}
```

### Type Expression

A type expression represents a literal MLIR type. It allows for statically
specifying an MLIR type to use, by specifying the textual form of that type.

```pdll
let i32Constant = op<arith.constant> -> (type<"i32">);
```

### Tuples

PDLL provides native support for tuples, which are used to group multiple
elements into a single compound value. The values in a tuple can be of any type,
and do not need to be of the same type. There is also no limit to the number of
elements held by a tuple. The elements of a tuple can be accessed by index:

```pdll
let tupleValue = (op<my_dialect.foo>, attr<"10 : i32">, type<"i32">);

let opValue = tupleValue.0;
let attrValue = tupleValue.1;
let typeValue = tupleValue.2;
```

You can also name the elements of a tuple and use those names to refer to the
values of the individual elements. An element name consists of an identifier
followed immediately by an equal (=).

```pdll
let tupleValue = (
  opValue = op<my_dialect.foo>,
  attr<"10 : i32">,
  typeValue = type<"i32">
);

let opValue = tupleValue.opValue;
let attrValue = tupleValue.1;
let typeValue = tupleValue.typeValue;
```

Tuples are used to represent multiple results from a
[constraint](#constraints-with-multiple-results) or
[rewrite](#rewrites-with-multiple-results).

### Constraints

Constraints provide the ability to inject additional checks on the input IR
within the `match` section of a pattern. Constraints can be applied anywhere
within the `match` section, and depending on the type can either be applied via
the constraint list of a [variable](#variables) or via the call operator (e.g.
`MyConstraint(...)`). There are three main categories of constraints:

#### Core Constraints

PDLL defines a number of core constraints that constrain the type of the IR
entity. These constraints can only be applied via the
[constraint list](#variable-constraints) of a variable.

*   `Attr` (`<` type `>`)?

A single entity constraint that corresponds to an `mlir::Attribute`. This
constraint optionally takes a type component that constrains the result type of
the attribute.

```pdll
// Define a simple variable using the `Attr` constraint.
let attr: Attr;
let constant = op<arith.constant> {value = attr};

// Define a simple variable using the `Attr` constraint, that has its type
// constrained as well.
let attrType: Type;
let attr: Attr<attrType>;
let constant = op<arith.constant> {value = attr};
```

*   `Op` (`<` op-name `>`)?

A single entity constraint that corresponds to an `mlir::Operation *`.

```pdll
// Match only when the input is from another operation.
let inputOp: Op;
let root = op<my_dialect.foo>(inputOp);

// Match only when the input is from another `my_dialect.foo` operation.
let inputOp: Op<my_dialect.foo>;
let root = op<my_dialect.foo>(inputOp);
```

*   `Type`

A single entity constraint that corresponds to an `mlir::Type`.

```pdll
// Define a simple variable using the `Type` constraint.
let resultType: Type;
let root = op<my_dialect.foo> -> (resultType);
```

*   `TypeRange`

A single entity constraint that corresponds to a `mlir::TypeRange`.

```pdll
// Define a simple variable using the `TypeRange` constraint.
let resultTypes: TypeRange;
let root = op<my_dialect.foo> -> (resultTypes);
```

*   `Value` (`<` type-expr `>`)?

A single entity constraint that corresponds to an `mlir::Value`. This constraint
optionally takes a type component that constrains the result type of the value.

```pdll
// Define a simple variable using the `Value` constraint.
let value: Value;
let root = op<my_dialect.foo>(value);

// Define a variable using the `Value` constraint, that has its type constrained
// to be same as the result type of the `root` op.
let valueType: Type;
let input: Value<valueType>;
let root = op<my_dialect.foo>(input) -> (valueType);
```

*   `ValueRange` (`<` type-expr `>`)?

A single entity constraint that corresponds to a `mlir::ValueRange`. This
constraint optionally takes a type component that constrains the result types of
the value range.

```pdll
// Define a simple variable using the `ValueRange` constraint.
let inputs: ValueRange;
let root = op<my_dialect.foo>(inputs);

// Define a variable using the `ValueRange` constraint, that has its types
// constrained to be same as the result types of the `root` op.
let valueTypes: TypeRange;
let inputs: ValueRange<valueTypes>;
let root = op<my_dialect.foo>(inputs) -> (valueTypes);
```

#### Defining Constraints in PDLL

Aside from the core constraints, additional constraints can also be defined
within PDLL. This allows for building matcher fragments that can be composed
across many different patterns. A constraint in PDLL is defined similarly to a
function in traditional programming languages; it contains a name, a set of
input arguments, a set of result types, and a body. Results of a constraint are
returned via a `return` statement. A few examples are shown below:

```pdll
/// A constraint that takes an input and constrains the use to an operation of
/// a given type.
Constraint UsedByFooOp(value: Value) {
  op<my_dialect.foo>(value);
}

/// A constraint that returns a result of an existing operation.
Constraint ExtractResult(op: Op<my_dialect.foo>) -> Value {
  return op.result;
}

Pattern {
  let value = ExtractResult(op<my_dialect.foo>);
  UsedByFooOp(value);
}
```

##### Constraints with multiple results

Constraints can return multiple results by returning a tuple of values. When
returning multiple results, each result can also be assigned a name to use when
indexing that tuple element. Tuple elements can be referenced by their index
number, or by name if they were assigned one.

```pdll
// A constraint that returns multiple results, with some of the results assigned
// a more readable name.
Constraint ExtractMultipleResults(op: Op<my_dialect.foo>) -> (Value, result1: Value) {
  return (op.result1, op.result2);
}

Pattern {
  // Return a tuple of values.
  let result = ExtractMultipleResults(op: op<my_dialect.foo>);

  // Index the tuple elements by index, or by name. 
  replace op<my_dialect.foo> with (result.0, result.1, result.result1);
}
```

##### Constraint result type inference

In addition to explicitly specifying the results of the constraint via the
constraint signature, PDLL defined constraints also support inferring the result
type from the return statement. Result type inference is active whenever the
constraint is defined with no result constraints:

```pdll
// This constraint returns a derived operation.
Constraint ReturnSelf(op: Op<my_dialect.foo>) {
  return op;
}
// This constraint returns a tuple of two Values.
Constraint ExtractMultipleResults(op: Op<my_dialect.foo>) {
  return (result1 = op.result1, result2 = op.result2);
}

Pattern {
  let values = ExtractMultipleResults(op<my_dialect.foo>);
  replace op<my_dialect.foo> with (values.result1, values.result2);
}
```

##### Single Line "Lambda" Body

Constraints generally define their body using a compound block of statements, as
shown below:

```pdll
Constraint ReturnSelf(op: Op<my_dialect.foo>) {
  return op;
}
Constraint ExtractMultipleResults(op: Op<my_dialect.foo>) {
  return (result1 = op.result1, result2 = op.result2);
}
```

Constraints also support a lambda-like syntax for specifying simple single line
bodies. The lambda body of a Constraint expects a single expression, which is
implicitly returned:

```pdll
Constraint ReturnSelf(op: Op<my_dialect.foo>) => op;

Constraint ExtractMultipleResults(op: Op<my_dialect.foo>)
  => (result1 = op.result1, result2 = op.result2);
```

#### Native Constraints

Constraints may also be defined outside of PDLL, and registered natively within
the C++ API.

##### Importing existing Native Constraints

Constraints defined externally can be imported into PDLL by specifying a
constraint "declaration". This is similar to the PDLL form of defining a
constraint but omits the body. Importing the declaration in this form allows for
PDLL to statically know the expected input and output types.

```pdll
// Import a single entity value native constraint that checks if the value has a
// single use. This constraint must be registered by the consumer of the
// compiled PDL.
Constraint HasOneUse(value: Value);

// Import a multi-entity type constraint that checks if two values have the same
// element type.
Constraint HasSameElementType(value1: Value, value2: Value);

Pattern {
  // A single entity constraint can be applied via the variable argument list.
  let value: HasOneUse;

  // Otherwise, constraints can be applied via the call operator:
  let value: Value = ...;
  let value2: Value = ...;
  HasOneUse(value);
  HasSameElementType(value, value2);
}
```

External constraints are those registered explicitly with the `RewritePatternSet` via
the C++ PDL API. For example, the constraints above may be registered as:

```c++
static LogicalResult hasOneUseImpl(PatternRewriter &rewriter, Value value) {
  return success(value.hasOneUse());
}
static LogicalResult hasSameElementTypeImpl(PatternRewriter &rewriter,
                                            Value value1, Value Value2) {
  return success(value1.getType().cast<ShapedType>().getElementType() ==
                 value2.getType().cast<ShapedType>().getElementType());
}

void registerNativeConstraints(RewritePatternSet &patterns) {
    patternList.getPDLPatterns().registerConstraintFunction(
        "HasOneUse", hasOneUseImpl);
    patternList.getPDLPatterns().registerConstraintFunction(
        "HasSameElementType", hasSameElementTypeImpl);
}
```

##### Defining Native Constraints in PDLL

In addition to importing native constraints, PDLL also supports defining native
constraints directly when compiling ahead-of-time (AOT) for C++. These
constraints can be defined by specifying a string code block after the
constraint declaration:

```pdll
Constraint HasOneUse(value: Value) [{
  return success(value.hasOneUse());
}];
Constraint HasSameElementType(value1: Value, value2: Value) [{
  return success(value1.getType().cast<ShapedType>().getElementType() ==
                 value2.getType().cast<ShapedType>().getElementType());
}];

Pattern {
  // A single entity constraint can be applied via the variable argument list.
  let value: HasOneUse;

  // Otherwise, constraints can be applied via the call operator:
  let value: Value = ...;
  let value2: Value = ...;
  HasOneUse(value);
  HasSameElementType(value, value2);
}
```

The arguments of the constraint are accessible within the code block via the
same name. See the ["type translation"](#native-constraint-type-translations) below for
detailed information on how PDLL types are converted to native types. In addition to the
PDLL arguments, the code block may also access the current `PatternRewriter` using
`rewriter`. The result type of the native constraint function is implicitly defined
as a `::mlir::LogicalResult`.

Taking the constraints defined above as an example, these function would roughly be
translated into:

```c++
LogicalResult HasOneUse(PatternRewriter &rewriter, Value value) {
  return success(value.hasOneUse());
}
LogicalResult HasSameElementType(Value value1, Value value2) {
  return success(value1.getType().cast<ShapedType>().getElementType() ==
                 value2.getType().cast<ShapedType>().getElementType());
}
```

TODO: Native constraints should also be allowed to return values in certain cases.

###### Native Constraint Type Translations

The types of argument and result variables are generally mapped to the corresponding
MLIR type of the [constraint](#constraints) used. Below is a detailed description
of how the mapped type of a variable is determined for the various different types of
constraints.

* Attr, Op, Type, TypeRange, Value, ValueRange:

These are all core constraints, and are mapped directly to the MLIR equivalent
(that their names suggest), namely:

  * `Attr`       -> "::mlir::Attribute"
  * `Op`         -> "::mlir::Operation *"
  * `Type`       -> "::mlir::Type"
  * `TypeRange`  -> "::mlir::TypeRange"
  * `Value`      -> "::mlir::Value"
  * `ValueRange` -> "::mlir::ValueRange"

* Op<dialect.name>

A named operation constraint has a unique translation. If the ODS registration of the
referenced operation has been included, the qualified C++ is used. If the ODS information
is not available, this constraint maps to "::mlir::Operation *", similarly to the unnamed
variant. For example, given the following:

```pdll
// `my_ops.td` provides the ODS definition of the `my_dialect` operations, such as
// `my_dialect.bar` used below.
#include "my_ops.td"

Constraint Cst(op: Op<my_dialect.bar>) [{
  return success(op ... );
}];
```

The native type used for `op` may be of the form `my_dialect::BarOp`, as opposed to the
default `::mlir::Operation *`. Below is a sample translation of the above constraint:

```c++
LogicalResult Cst(my_dialect::BarOp op) {
  return success(op ... );
}
```

* Imported ODS Constraints

Aside from the core constraints, certain constraints imported from ODS may use a unique
native type. How to enable this unique type depends on the ODS constraint construct that
was imported:

  * `Attr` constraints
    - Imported `Attr` constraints utilize the `storageType` field for native type translation.
  
  * `Type` constraints
    - Imported `Type` constraints utilize the `cppClassName` field for native type translation.

  * `AttrInterface`/`OpInterface`/`TypeInterface` constraints
    - Imported interfaces utilize the `cppInterfaceName` field for native type translation.

#### Defining Constraints Inline

In addition to global scope, PDLL Constraints and Native Constraints defined in
PDLL may be specified *inline* at any level of nesting. This means that they may
be defined in Patterns, other Constraints, Rewrites, etc:

```pdll
Constraint GlobalConstraint() {
  Constraint LocalConstraint(value: Value) {
    ...
  };
  Constraint LocalNativeConstraint(value: Value) [{
    ...
  }];
  let someValue: [LocalConstraint, LocalNativeConstraint] = ...;
}
```

Constraints that are defined inline may also elide the name when used directly:

```pdll
Constraint GlobalConstraint(inputValue: Value) {
  Constraint(value: Value) { ... }(inputValue);
  Constraint(value: Value) [{ ... }](inputValue);
}
```

When defined inline, PDLL constraints may reference any previously defined
variable:

```pdll
Constraint GlobalConstraint(op: Op<my_dialect.foo>) {
  Constraint LocalConstraint() {
    let results = op.results;
  };
}
```

### Rewriters

Rewriters define the set of transformations to be performed within the `rewrite`
section of a pattern, and, more specifically, how to transform the input IR
after a successful pattern match. All PDLL rewrites must be defined within the
`rewrite` section of the pattern. The `rewrite` section is denoted by the last
statement within the body of the `Pattern`, which is required to be an
[operation rewrite statement](#operation-rewrite-statements). There are two main
categories of rewrites in PDLL: operation rewrite statements, and user defined
rewrites.

#### Operation Rewrite statements

Operation rewrite statements are builtin PDLL statements that perform an IR
transformation given a root operation. These statements are the only ones able
to start the `rewrite` section of a pattern, as they allow for properly
["binding"](#variable-binding) the root operation of the pattern.

##### `erase` statement

```pdll
// A pattern that erases all `my_dialect.foo` operations.
Pattern => erase op<my_dialect.foo>;
```

The `erase` statement erases a given operation.

##### `replace` statement

```pdll
// A pattern that replaces the root operation with its input value.
Pattern {
  let root = op<my_dialect.foo>(input: Value);
  replace root with input;
}

// A pattern that replaces the root operation with multiple input values.
Pattern {
  let root = op<my_dialect.foo>(input: Value, _: Value, input2: Value);
  replace root with (input, input2);
}

// A pattern that replaces the root operation with another operation.
// Note that when an operation is used as the replacement, we can infer its
// result types from the input operation. In these cases, the result
// types of replacement operation may be elided. 
Pattern {
  // Note: In this pattern we also inlined the `root` expression.
  replace op<my_dialect.foo> with op<my_dialect.bar>;
}
```

The `replace` statement allows for replacing a given root operation with either
another operation, or a set of input `Value` and `ValueRange` values. When an operation
is used as the replacement, we allow infering the result types from the input operation.
In these cases, the result types of replacement operation may be elided. Note that no
other components aside from the result types will be inferred from the input operation
during the replacement.

##### `rewrite` statement

```pdll
// A simple pattern that replaces the root operation with its input value.
Pattern {
  let root = op<my_dialect.foo>(input: Value);
  rewrite root with {
    ...

    replace root with input;
  };
}
```

The `rewrite` statement allows for rewriting a given root operation with a block
of nested rewriters. The root operation is not implicitly erased or replaced,
and any transformations to it must be expressed within the nested rewrite block.
The inner body may contain any number of other rewrite statements, variables, or
expressions.

#### Defining Rewriters in PDLL

Additional rewrites can also be defined within PDLL, which allows for building
rewrite fragments that can be composed across many different patterns. A
rewriter in PDLL is defined similarly to a function in traditional programming
languages; it contains a name, a set of input arguments, a set of result types,
and a body. Results of a rewrite are returned via a `return` statement. A few
examples are shown below:

```pdll
// A rewrite that constructs and returns a new operation, given an input value.
Rewrite BuildFooOp(value: Value) -> Op {
  return op<my_dialect.foo>(value);
}

Pattern {
  // We invoke the rewrite in the same way as functions in traditional
  // languages.
  replace op<my_dialect.old_op>(input: Value) with BuildFooOp(input);
}
```

##### Rewrites with multiple results

Rewrites can return multiple results by returning a tuple of values. When
returning multiple results, each result can also be assigned a name to use when
indexing that tuple element. Tuple elements can be referenced by their index
number, or by name if they were assigned one.

```pdll
// A rewrite that returns multiple results, with some of the results assigned
// a more readable name.
Rewrite CreateRewriteOps() -> (Op, result1: ValueRange) {
  return (op<my_dialect.bar>, op<my_dialect.foo>);
}

Pattern {
  rewrite root: Op<my_dialect.foo> with {
    // Invoke the rewrite, which returns a tuple of values.
    let result = CreateRewriteOps();

    // Index the tuple elements by index, or by name. 
    replace root with (result.0, result.1, result.result1);
  }
}
```

##### Rewrite result type inference

In addition to explicitly specifying the results of the rewrite via the rewrite
signature, PDLL defined rewrites also support inferring the result type from the
return statement. Result type inference is active whenever the rewrite is
defined with no result constraints:

```pdll
// This rewrite returns a derived operation.
Rewrite ReturnSelf(op: Op<my_dialect.foo>) => op;
// This rewrite returns a tuple of two Values.
Rewrite ExtractMultipleResults(op: Op<my_dialect.foo>) {
  return (result1 = op.result1, result2 = op.result2);
}

Pattern {
  rewrite root: Op<my_dialect.foo> with {
    let values = ExtractMultipleResults(op<my_dialect.foo>);
    replace root with (values.result1, values.result2);
  }
}
```

##### Single Line "Lambda" Body

Rewrites generally define their body using a compound block of statements, as
shown below:

```pdll
Rewrite ReturnSelf(op: Op<my_dialect.foo>) {
  return op;
}
Rewrite EraseOp(op: Op) {
  erase op;
}
```

Rewrites also support a lambda-like syntax for specifying simple single line
bodies. The lambda body of a Rewrite expects a single expression, which is
implicitly returned, or a single
[operation rewrite statement](#operation-rewrite-statements):

```pdll
Rewrite ReturnSelf(op: Op<my_dialect.foo>) => op;
Rewrite EraseOp(op: Op) => erase op;
```

#### Native Rewriters

Rewriters may also be defined outside of PDLL, and registered natively within
the C++ API.

##### Importing existing Native Rewrites

Rewrites defined externally can be imported into PDLL by specifying a
rewrite "declaration". This is similar to the PDLL form of defining a
rewrite but omits the body. Importing the declaration in this form allows for
PDLL to statically know the expected input and output types.

```pdll
// Import a single input native rewrite that returns a new operation. This
// rewrite must be registered by the consumer of the compiled PDL.
Rewrite BuildOp(value: Value) -> Op;

Pattern {
  replace op<my_dialect.old_op>(input: Value) with BuildOp(input);
}
```

External rewrites are those registered explicitly with the `RewritePatternSet` via
the C++ PDL API. For example, the rewrite above may be registered as:

```c++
static Operation *buildOpImpl(PDLResultList &results, Value value) {
  // insert special rewrite logic here.
  Operation *resultOp = ...; 
  return resultOp;
}

void registerNativeRewrite(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("BuildOp", buildOpImpl);
}
```

##### Defining Native Rewrites in PDLL

In addition to importing native rewrites, PDLL also supports defining native
rewrites directly when compiling ahead-of-time (AOT) for C++. These rewrites can
be defined by specifying a string code block after the rewrite declaration:

```pdll
Rewrite BuildOp(value: Value) -> (foo: Op<my_dialect.foo>, bar: Op<my_dialect.bar>) [{
  return {rewriter.create<my_dialect::FooOp>(value), rewriter.create<my_dialect::BarOp>()};
}];

Pattern {
  let root = op<my_dialect.foo>(input: Value);
  rewrite root with {
    // Invoke the native rewrite and use the results when replacing the root.
    let results = BuildOp(input);
    replace root with (results.foo, results.bar);
  }
}
```

The arguments of the rewrite are accessible within the code block via the
same name. See the ["type translation"](#native-rewrite-type-translations) below for
detailed information on how PDLL types are converted to native types. In addition to the
PDLL arguments, the code block may also access the current `PatternRewriter` using
`rewriter`. See the ["result translation"](#native-rewrite-result-translation) section
for detailed information on how the result type of the native function is determined.

Taking the rewrite defined above as an example, this function would roughly be
translated into:

```c++
std::tuple<my_dialect::FooOp, my_dialect::BarOp> BuildOp(Value value) {
  return {rewriter.create<my_dialect::FooOp>(value), rewriter.create<my_dialect::BarOp>()};
}
```

###### Native Rewrite Type Translations

The types of argument and result variables are generally mapped to the corresponding
MLIR type of the [constraint](#constraints) used. The rules of native `Rewrite` type translation
are identical to those of native `Constraint`s, please view the corresponding
[native `Constraint` type translation](#native-constraint-type-translations) section for a
detailed description of how the mapped type of a variable is determined.

###### Native Rewrite Result Translation

The results of a native rewrite are directly translated to the results of the native function,
using the type translation rules [described above](#native-rewrite-type-translations). The section
below describes the various result translation scenarios:

* Zero Result

```pdll
Rewrite createOp() [{
  rewriter.create<my_dialect::FooOp>();
}];
```

In the case where a native `Rewrite` has no results, the native function returns `void`:

```c++
void createOp(PatternRewriter &rewriter) {
  rewriter.create<my_dialect::FooOp>();
}
```

* Single Result

```pdll
Rewrite createOp() -> Op<my_dialect.foo> [{
  return rewriter.create<my_dialect::FooOp>();
}];
```

In the case where a native `Rewrite` has a single result, the native function returns the corresponding
native type for that single result:

```c++
my_dialect::FooOp createOp(PatternRewriter &rewriter) {
  return rewriter.create<my_dialect::FooOp>();
}
```

* Multi Result

```pdll
Rewrite complexRewrite(value: Value) -> (Op<my_dialect.foo>, FunctionOpInterface) [{
  ...
}];
```

In the case where a native `Rewrite` has multiple results, the native function returns a `std::tuple<...>`
containing the corresponding native types for each of the results:

```c++
std::tuple<my_dialect::FooOp, FunctionOpInterface>
complexRewrite(PatternRewriter &rewriter, Value value) {
  ...
}
```

#### Defining Rewrites Inline

In addition to global scope, PDLL Rewrites and Native Rewrites defined in PDLL
may be specified *inline* at any level of nesting. This means that they may be
defined in Patterns, other Rewrites, etc:

```pdll
Rewrite GlobalRewrite(inputValue: Value) {
  Rewrite localRewrite(value: Value) {
    ...
  };
  Rewrite localNativeRewrite(value: Value) [{
    ...
  }];
  localRewrite(inputValue);
  localNativeRewrite(inputValue);
}
```

Rewrites that are defined inline may also elide the name when used directly:

```pdll
Rewrite GlobalRewrite(inputValue: Value) {
  Rewrite(value: Value) { ... }(inputValue);
  Rewrite(value: Value) [{ ... }](inputValue);
}
```

When defined inline, PDLL rewrites may reference any previously defined
variable:

```pdll
Rewrite GlobalRewrite(op: Op<my_dialect.foo>) {
  Rewrite localRewrite() {
    let results = op.results;
  };
}
```
