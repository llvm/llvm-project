# Table-driven Declarative Rewrite Rule (DRR)

In addition to subclassing the `mlir::RewritePattern` C++ class, MLIR also
supports defining rewrite rules in a declarative manner. Similar to
[Op Definition Specification](OpDefinitions.md) (ODS), this is achieved via
[TableGen][TableGen], which is a language to maintain records of domain-specific
information. The rewrite rules are specified concisely in a TableGen record,
which will be expanded into an equivalent `mlir::RewritePattern` subclass at
compiler build time.

This manual explains in detail all of the available mechanisms for defining
rewrite rules in such a declarative manner. It aims to be a specification
instead of a tutorial. Please refer to
[Quickstart tutorial to adding MLIR graph rewrite](Tutorials/QuickstartRewrites.md)
for the latter.

Given that declarative rewrite rules depend on op definition specification, this
manual assumes knowledge of the [ODS](OpDefinitions.md) doc.

## Benefits

Compared to the hand-written C++ classes, this declarative approach has several
benefits, including but not limited to:

*   **Being declarative**: The pattern creator just needs to state the rewrite
    pattern declaratively, without worrying about the concrete C++ methods to
    call.
*   **Removing boilerplate and showing the very essence of the rewrite**:
    `mlir::RewritePattern` is already good at hiding boilerplate for defining a
    rewrite rule. But we still need to write the class and function structures
    required by the C++ programming language, inspect ops for matching, and call
    op `build()` methods for constructing. These statements are typically quite
    simple and similar, so they can be further condensed with auto-generation.
    Because we reduce the boilerplate to the bare minimum, the declarative
    rewrite rule will just contain the very essence of the rewrite. This makes
    it very easy to understand the pattern.

## Strengths and Limitations

The declarative rewrite rule is **operation-based**: it describes a rule to
match against a directed acyclic graph (DAG) of operations and generate DAGs of
operations. This gives DRR both its strengths and limitations: it is good at
expressing op to op conversions, but not that well suited for, say, converting
an op into a loop nest.

Per the current implementation, DRR does not have good support for the following
features:

*   Matching and generating ops with regions.
*   Matching and generating ops with block arguments.
*   Matching multi-result ops in nested patterns.
*   Matching and generating variadic operand/result ops in nested patterns.
*   Packing and unpacking variadic operands/results during generation.
*   [`NativeCodeCall`](#nativecodecall-transforming-the-generated-op) returning
    more than one results.

## Rule Definition

The core construct for defining a rewrite rule is defined in
[`OpBase.td`][OpBase] as

```tablegen
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```

A declarative rewrite rule contains two main components:

*   A _source pattern_, which is used for matching a DAG of operations.
*   One or more _result patterns_, which are used for generating DAGs of
    operations to replace the matched DAG of operations.

We allow multiple result patterns to support
[multi-result ops](#supporting-multi-result-ops) and
[auxiliary ops](#supporting-auxiliary-ops), but frequently we just want to
convert one DAG of operations to another DAG of operations. There is a handy
wrapper of `Pattern`, `Pat`, which takes a single result pattern:

```tablegen
class Pat<
    dag sourcePattern, dag resultPattern,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)> :
  Pattern<sourcePattern, [resultPattern], additionalConstraints, benefitAdded>;
```

Each pattern is specified as a TableGen `dag` object with the syntax of
`(operator arg0, arg1, ...)`.

`operator` is typically an MLIR op, but it can also be other
[directives](#rewrite-directives). `argN` is for matching (if used in source
pattern) or generating (if used in result pattern) the `N`-th argument for
`operator`. If the `operator` is some MLIR operation, it means the `N`-th
argument as specified in the `arguments` list of the op's definition. Therefore,
we say op argument specification in pattern is **position-based**: the position
where they appear matters.

`argN` can be a `dag` object itself, thus we can have nested `dag` tree to model
the def-use relationship between ops.

### Source pattern

The source pattern is for matching a DAG of operations. Arguments in the `dag`
object are intended to **capture** the op arguments. They can also be used to
**further limit** the match criteria. The capturing is done by specifying a
symbol starting with the `$` sign, while further constraints are introduced by
specifying a `TypeConstraint` (for an operand) or a `AttrConstraint` (for an
attribute).

#### Binding op arguments and limiting the match

For example,

```tablegen
def AOp : Op<"a_op"> {
    let arguments = (ins
      AnyType:$a_input,
      AnyAttr:$a_attr
    );

    let results = (outs
      AnyType:$a_output
    );
}

def : Pat<(AOp $input, F32Attr:$attr), ...>;
```

In the above, we are matching an `AOp` whose `$input` can be anything valid as
defined by the op and whose `$attr` must be a float attribute. If the match
succeeds, we bind the `$input` symbol to the op's only input (`$a_input`) and
`$attr` to the only attribute (`$a_attr`); we can reference them using `$input`
and `$attr` in result patterns and additional constraints.

The pattern is position-based: the symbol names used for capturing here do not
need to match with the op definition as shown in the above example. As another
example, the pattern can be written as `def : Pat<(AOp $a, F32Attr:$b), ...>;`
and use `$a` and `$b` to refer to the captured input and attribute. But using
the ODS name directly in the pattern is also allowed. Operands in the source
pattern can have the same name. This bounds one operand to the name while
verifying the rest are all equal.

Also note that we only need to add `TypeConstraint` or `AttributeConstraint`
when we need to further limit the match criteria. If all valid cases to the op
are acceptable, then we can leave the constraint unspecified.

`$_` is a special symbol to mean ignore capturing an argument. For example,
`def : Pat<(AOp $_, $b), ...>` means only `$b` is interesting to capture and
will be referenced later in result patterns. It's still possible to place
additional constraints even if the symbol is not to be captured; for such case,
you can simply use just the `TypeConstraint` or `AttributeConstraint` without a
bound symbol, for example, `def : Pat<(AOp $a, F32Attr), ...>`.

#### Matching DAG of operations

To match a DAG of ops, use nested `dag` objects:

```tablegen

def BOp : Op<"b_op"> {
    let arguments = (ins);

    let results = (outs
      AnyType:$b_output
    );
}


def : Pat<(AOp (BOp), $attr), ...>;
```

The above pattern matches an `AOp` whose only operand is generated by a `BOp`,
that is, the following MLIR code:

```mlir
%0 = "b_op"() : () -> (...)
%1 = "a_op"(%0) {attr: ...} : () -> (...)
```

#### Binding op results

To bind a symbol to the results of a matched op for later reference, attach the
symbol to the op itself:

```tablegen
def : Pat<(AOp (BOp:$b_result), $attr), ...>;
```

The above will bind `$b_result` to the matched `BOp`'s result. (There are more
details regarding multi-result ops, which is covered
[later](#supporting-multi-result-ops).)

### Result pattern

The result pattern is for generating a DAG of operations. Arguments in the `dag`
object are intended to **reference** values captured in the source pattern and
potentially **apply transformations**.

#### Referencing bound symbols

For example,

```tablegen
def COp : Op<"c_op"> {
    let arguments = (ins
      AnyType:$c_input,
      AnyAttr:$c_attr
    );

    let results = (outs
      AnyType:$c_output
    );
}

def : Pat<(AOp $input, $attr), (COp $input, $attr)>;
```

In the above, `AOp`'s only operand and attribute are bound to `$input` and
`$attr`, respectively. We then reference them in the result pattern for
generating the `COp` by passing them in as arguments to `COp`'s `build()`
method.

We can also reference symbols bound to matched op's results:

```tablegen
def : Pat<(AOp (BOp:$b_result) $attr), (COp $b_result $attr)>;
```

In the above, we are using `BOp`'s result for building `COp`.

#### Building operations

Given that `COp` was specified with table-driven op definition, there will be
several `build()` methods generated for it. One of them has aggregated
parameters for result types, operands, and attributes in the signature: `void
COp::build(..., ArrayRef<Type> resultTypes, Array<Value> operands,
ArrayRef<NamedAttribute> attr)`. The pattern in the above calls this `build()`
method for constructing the `COp`.

In general, arguments in the result pattern will be passed directly to the
`build()` method to leverage the auto-generated `build()` method, list them in
the pattern by following the exact same order as the ODS `arguments` definition.
Otherwise, a custom `build()` method that matches the argument list is required.

Right now all ODS-generated `build()` methods require specifying the result
type(s), unless the op has known traits like `SameOperandsAndResultType` that we
can use to auto-generate a `build()` method with result type deduction. When
generating an op to replace the result of the matched root op, we can use the
matched root op's result type when calling the ODS-generated builder. Otherwise
(e.g., generating an [auxiliary op](#supporting-auxiliary-ops) or generating an
op with a nested result pattern), DRR will not be able to deduce the result
type(s). The pattern author will need to define a custom builder that has result
type deduction ability via `OpBuilder` in ODS. For example, in the following
pattern

```tablegen
def : Pat<(AOp $input, $attr), (COp (AOp $input, $attr) $attr)>;
```

`AOp` is generated via a nested result pattern; DRR won't be able to deduce the
result type for it. A custom builder for `AOp` should be defined and it should
deduce the result type by itself. The builder should have the separate parameter
for each operand and attribute and deduce the result type internally by itself.
For example, for the above `AOp`, a possible builder is:

```c++

void AOp::build(OpBuilder &builder, OperationState &state,
                Value input, Attribute attr) {
  state.addOperands({input});
  state.addAttribute("a_attr", attr);
  Type type = ...; // Deduce result type here
  state.addTypes({type});
}
```

Failing to define such a builder will result in an error at C++ compilation time
saying the call to `AOp::build()` cannot be resolved because of the number of
parameters mismatch.

#### Generating DAG of operations

`dag` objects can be nested to generate a DAG of operations:

```tablegen
def : Pat<(AOp $input, $attr), (COp (BOp), $attr)>;
```

In the above, we generate a `BOp`, and then use its result to generate the `COp`
to replace the matched `AOp`.

#### Binding op results

In the result pattern, we can bind to the result(s) of a newly built op by
attaching symbols to the op. (But we **cannot** bind to op arguments given that
they are referencing previously bound symbols.) This is useful for reusing newly
created results where suitable. For example,

```tablegen
def DOp : Op<"d_op"> {
    let arguments = (ins
      AnyType:$d_input1,
      AnyType:$d_input2,
    );

    let results = (outs
      AnyType:$d_output
    );
}

def : Pat<(AOp $input, $ignored_attr), (DOp (BOp:$b_result) $b_result)>;
```

In this pattern, an `AOp` is matched and replaced with a `DOp` whose two
operands are from the result of a single `BOp`. This is only possible by binding
the result of the `BOp` to a name and reuse it for the second operand of the
`DOp`

#### `NativeCodeCall`: transforming the generated op

Sometimes the captured arguments are not exactly what we want so they cannot be
directly fed in as arguments to build the new op. For such cases, we can apply
transformations on the arguments by calling into C++ helper functions. This is
achieved by `NativeCodeCall`.

For example, if we want to capture some op's attributes and group them as an
array attribute to construct a new op:

```tablegen

def TwoAttrOp : Op<"two_attr_op"> {
    let arguments = (ins
      AnyAttr:$op_attr1,
      AnyAttr:$op_attr2
    );

    let results = (outs
      AnyType:$op_output
    );
}

def OneAttrOp : Op<"one_attr_op"> {
    let arguments = (ins
      ArrayAttr:$op_attr
    );

    let results = (outs
      AnyType:$op_output
    );
}
```

We can write a C++ helper function:

```c++
ArrayAttr createArrayAttr(Builder &builder, Attribute a, Attribute b) {
  return builder.getArrayAttr({a, b});
}
```

And then write the pattern as:

```tablegen
def createArrayAttr : NativeCodeCall<"createArrayAttr($_builder, $0, $1)">;

def : Pat<(TwoAttrOp $attr1, $attr2),
          (OneAttrOp (createArrayAttr $attr1, $attr2))>;
```

And make sure the generated C++ code from the above pattern has access to the
definition of the C++ helper function.

In the above example, we are using a string to specialize the `NativeCodeCall`
template. The string can be an arbitrary C++ expression that evaluates into some
C++ object expected at the `NativeCodeCall` site (here it would be expecting an
array attribute). Typically the string should be a function call.

##### `NativeCodeCall` placeholders

In `NativeCodeCall`, we can use placeholders like `$_builder`, `$N` and `$N...`.
The former is called _special placeholder_, while the latter is called
_positional placeholder_ and _positional range placeholder_.

`NativeCodeCall` right now only supports three special placeholders:
`$_builder`, `$_loc`, and `$_self`:

*   `$_builder` will be replaced by the current `mlir::PatternRewriter`.
*   `$_loc` will be replaced by the fused location or custom location (as
    determined by location directive).
*   `$_self` will be replaced by the defining operation in a source pattern.

We have seen how `$_builder` can be used in the above; it allows us to pass a
`mlir::Builder` (`mlir::PatternRewriter` is a subclass of `mlir::OpBuilder`,
which is a subclass of `mlir::Builder`) to the C++ helper function to use the
handy methods on `mlir::Builder`.

Here's an example how we should use `$_self` in source pattern,

```tablegen

def : Pat<(OneAttrOp (NativeCodeCall<"Foo($_self, &$0)"> I32Attr:$val)),
          (TwoAttrOp $val, $val)>;
```

In the above, `$_self` is substituted by the defining operation of the first
operand of OneAttrOp. Note that we don't support binding name to `NativeCodeCall`
in the source pattern. To carry some return values from a helper function, put the
names (constraint is optional) in the parameter list and they will be bound to
the variables with correspoding type. Then these names must be either passed by
reference or pointer to the variable used as argument so that the matched value
can be returned. In the same example, `$val` will be bound to a variable with
`Attribute` type (as `I32Attr`) and the type of the second argument in `Foo()`
could be `Attribute&` or `Attribute*`. Names with attribute constraints will be
captured as `Attribute`s while everything else will be treated as `Value`s.

Positional placeholders will be substituted by the `dag` object parameters at
the `NativeCodeCall` use site. For example, if we define `SomeCall :
NativeCodeCall<"someFn($1, $2, $0)">` and use it like `(SomeCall $in0, $in1,
$in2)`, then this will be translated into C++ call `someFn($in1, $in2, $in0)`.

Positional range placeholders will be substituted by multiple `dag` object
parameters at the `NativeCodeCall` use site. For example, if we define
`SomeCall : NativeCodeCall<"someFn($1...)">` and use it like `(SomeCall $in0,
$in1, $in2)`, then this will be translated into C++ call `someFn($in1, $in2)`.

##### `NativeCodeCall` binding multi-results

To bind multi-results and access the N-th result with `$<name>__N`, specify the
number of return values in the template. Note that only `Value` type is
supported for multiple results binding. For example,

```tablegen

def PackAttrs : NativeCodeCall<"packAttrs($0, $1)", 2>;
def : Pattern<(TwoResultOp $attr1, $attr2),
              [(OneResultOp (PackAttr:$res__0, $attr1, $attr2)),
               (OneResultOp $res__1)]>;

```

Use `NativeCodeCallVoid` for cases with no return value.

The correct number of returned value specified in NativeCodeCall is important.
It will be used to verify the consistency of the number of return values.
Additionally, `mlir-tblgen` will try to capture the return values of
`NativeCodeCall` in the generated code so that it will trigger a later compilation
error if a `NativeCodeCall` that doesn't return any result isn't labeled with 0
returns.

##### Customizing entire op building

`NativeCodeCall` is not only limited to transforming arguments for building an
op; it can be also used to specify how to build an op entirely. An example:

If we have a C++ function for building an op:

```c++
Operation *createMyOp(OpBuilder builder, Value input, Attribute attr);
```

We can wrap it up and invoke it like:

```tablegen
def createMyOp : NativeCodeCall<"createMyOp($_builder, $0, $1)">;

def : Pat<(... $input, $attr), (createMyOp $input, $attr)>;
```

### Supporting auxiliary ops

A declarative rewrite rule supports multiple result patterns. One of the
purposes is to allow generating _auxiliary ops_. Auxiliary ops are operations
used for building the replacement ops; but they are not directly used for
replacement themselves.

For the case of uni-result ops, if there are multiple result patterns, only the
value generated from the last result pattern will be used to replace the matched
root op's result; all other result patterns will be considered as generating
auxiliary ops.

Normally we want to specify ops as nested `dag` objects if their def-use
relationship can be expressed in the way that an op's result can feed as the
argument to consuming op. But that is not always possible. For example, if we
want to allocate memory and store some computation (in pseudocode):

```mlir
%dst = addi %lhs, %rhs
```

into

```mlir
%shape = shape %lhs
%mem = alloc %shape
%sum = addi %lhs, %rhs
store %mem, %sum
%dst = load %mem
```

We cannot fit in with just one result pattern given `store` does not return a
value. Instead we can use multiple result patterns:

```tablegen
def : Pattern<(AddIOp $lhs, $rhs),
              [(StoreOp (AllocOp:$mem (ShapeOp $lhs)), (AddIOp $lhs, $rhs)),
               (LoadOp $mem)];
```

In the above we use the first result pattern to generate the first four ops, and
use the last pattern to generate the last op, which is used to replace the
matched op.

### Supporting multi-result ops

Multi-result ops bring extra complexity to declarative rewrite rules. We use
TableGen `dag` objects to represent ops in patterns; there is no native way to
indicate that an op generates multiple results. The approach adopted is based on
**naming convention**: a `__N` suffix is added to a symbol to indicate the
`N`-th result.

#### `__N` suffix

The `__N` suffix is specifying the `N`-th result as a whole (which can be
[variadic](#supporting-variadic-ops)). For example, we can bind a symbol to some
multi-result op and reference a specific result later:

```tablegen
def ThreeResultOp : Op<"three_result_op"> {
    let arguments = (ins ...);

    let results = (outs
      AnyTensor:$output1,
      AnyTensor:$output2,
      AnyTensor:$output3
    );
}

def : Pattern<(ThreeResultOp:$results ...),
              [(... $results__0), ..., (... $results__2), ...]>;
```

In the above pattern we bind `$results` to all the results generated by
`ThreeResultOp` and references its `$output1` and `$output3` later in the result
patterns.

We can also bind a symbol and reference one of its specific result at the same
time, which is typically useful when generating multi-result ops:

```tablegen
// TwoResultOp has similar definition as ThreeResultOp, but only has two
// results.

def : Pattern<(TwoResultOp ...),
              [(ThreeResultOp:$results__2, ...),
               (replaceWithValue $results__0)]>;
```

In the above, we created a `ThreeResultOp` and bind `results` to its results,
and uses its last result (`$output3`) and first result (`$output1`) to replace
the `TwoResultOp`'s two results, respectively.

#### Replacing multi-result ops

The above example also shows how to replace a matched multi-result op.

To replace an `N`-result op, the result patterns must generate at least `N`
declared values (see [Declared vs. actual value](#declared-vs-actual-value) for
definition). If there are more than `N` declared values generated, only the last
`N` declared values will be used to replace the matched op. Note that because of
the existence of multi-result op, one result pattern **may** generate multiple
declared values. So it means we do not necessarily need `N` result patterns to
replace an `N`-result op. For example, to replace an op with three results, you
can have

```tablegen
// ThreeResultOp/TwoResultOp/OneResultOp generates three/two/one result(s),
// respectively.

// Replace each result with a result generated from an individual op.
def : Pattern<(ThreeResultOp ...),
              [(OneResultOp ...), (OneResultOp ...), (OneResultOp ...)]>;

// Replace the first two results with two results generated from the same op.
def : Pattern<(ThreeResultOp ...),
              [(TwoResultOp ...), (OneResultOp ...)]>;

// Replace all three results with three results generated from the same op.
def : Pat<(ThreeResultOp ...), (ThreeResultOp ...)>;

def : Pattern<(ThreeResultOp ...),
              [(AuxiliaryOp ...), (ThreeResultOp ...)]>;
```

But using a single op to serve as both auxiliary op and replacement op is
forbidden, i.e., the following is not allowed because that the first
`TwoResultOp` generates two results but only the second result is used for
replacing the matched op's result:

```tablegen
def : Pattern<(ThreeResultOp ...),
              [(TwoResultOp ...), (TwoResultOp ...)]>;
```

### Supporting variadic ops

#### Declared vs. actual value

Before going into details on variadic op support, we need to define a few terms
regarding an op's values.

*   _Value_: either an operand or a result
*   _Declared operand/result/value_: an operand/result/value statically declared
    in ODS of the op
*   _Actual operand/result/value_: an operand/result/value of an op instance at
    runtime

The above terms are needed because ops can have multiple results, and some of
the results can also be variadic. For example,

```tablegen
def MultiVariadicOp : Op<"multi_variadic_op"> {
    let arguments = (ins
      AnyTensor:$input1,
      Variadic<AnyTensor>:$input2,
      AnyTensor:$input3
    );

    let results = (outs
      AnyTensor:$output1,
      Variadic<AnyTensor>:$output2,
      AnyTensor:$output3
    );
}
```

We say the above op has 3 declared operands and 3 declared results. But at
runtime, an instance can have 3 values corresponding to `$input2` and 2 values
correspond to `$output2`; we say it has 5 actual operands and 4 actual results.
A variadic operand/result is a considered as a declared value that can
correspond to multiple actual values.

[TODO]

### Supplying additional constraints

Constraints can be placed on op arguments when matching. But sometimes we need
to also place constraints on the matched op's results or sometimes need to limit
the matching with some constraints that cover both the arguments and the
results. The third parameter to `Pattern` (and `Pat`) is for this purpose.

For example, we can write

```tablegen
def HasNoUseOf: Constraint<CPred<"$_self.use_empty()">, "has no use">;

def HasSameElementType : Constraint<
    CPred<"$0.cast<ShapedType>().getElementType() == "
          "$1.cast<ShapedType>().getElementType()">,
    "has same element type">;

def : Pattern<(TwoResultOp:$results $input),
              [(...), (...)],
              [(F32Tensor:$results__0), (HasNoUseOf:$results__1),
               (HasSameElementShape $results__0, $input)]>;
```

You can

*   Use normal `TypeConstraint`s on previous bound symbols (the first result of
    `TwoResultOp` must be a float tensor);
*   Define new `Constraint` for previous bound symbols (the second result of
    `TwoResultOp` must has no use);
*   Apply constraints on multiple bound symbols (`$input` and `TwoResultOp`'s
    first result must have the same element type).

### Adjusting benefits

The benefit of a `Pattern` is an integer value indicating the benefit of
matching the pattern. It determines the priorities of patterns inside the
pattern rewrite driver. A pattern with a higher benefit is applied before one
with a lower benefit.

In DRR, a rule is set to have a benefit of the number of ops in the source
pattern. This is based on the heuristics and assumptions that:

*   Larger matches are more beneficial than smaller ones.
*   If a smaller one is applied first the larger one may not apply anymore.

The fourth parameter to `Pattern` (and `Pat`) allows to manually tweak a
pattern's benefit. Just supply `(addBenefit N)` to add `N` to the benefit value.

## Rewrite directives

### `location`

By default the C++ pattern expanded from a DRR pattern uses the fused location
of all source ops as the location for all generated ops. This is not always the
best location mapping relationship. For such cases, DRR provides the `location`
directive to provide finer control.

`location` is of the following syntax:

```tablegen
(location $symbol0, $symbol1, ...)
```

where all `$symbol` should be bound previously in the pattern and one optional
string may be specified as an attribute. The following locations are created:

*   If only 1 symbol is specified then that symbol's location is used,
*   If multiple are specified then a fused location is created;
*   If no symbol is specified then string must be specified and a NamedLoc is
    created instead;

`location` must be used as a trailing argument to an op creation. For example,

```tablegen
def : Pat<(LocSrc1Op:$src1 (LocSrc2Op:$src2 ...),
          (LocDst1Op (LocDst2Op ..., (location $src2)), (location "outer"))>;
```

In the above pattern, the generated `LocDst2Op` will use the matched location of
`LocSrc2Op` while the root `LocDst1Op` node will used the named location
`outer`.

### `replaceWithValue`

The `replaceWithValue` directive is used to eliminate a matched op by replacing
all of its uses with a captured value. It is of the following syntax:

```tablegen
(replaceWithValue $symbol)
```

where `$symbol` should be a symbol bound previously in the pattern.

For example,

```tablegen
def : Pat<(Foo $input), (replaceWithValue $input)>;
```

The above pattern removes the `Foo` and replaces all uses of `Foo` with
`$input`.

### `returnType`

The `returnType` directive allows patterns to directly specify return types for
replacement ops that lack return type inference with op traits or user-defined
builders with return type deduction.

The `returnType` directive must be used as a trailing argument to a node
describing a replacement op. The directive comes in three forms:

* `(returnType $value)`: copy the type of the operand or result bound to
  `value`.
* `(returnType "$_builder.getI32Type()")`: a string literal embedding C++. The
  embedded snippet is expected to return a `Type` or a `TypeRange`.
* `(returnType (NativeCodeCall<"myFunc($0)"> $value))`: a DAG node with a native
  code call that can be passed any bound variables arguments.

Specify multiple return types with a mix of any of the above. Example:

```tablegen
def : Pat<(SourceOp $arg0, $arg1),
          (OpA $arg0, (TwoResultOp:$res__1 $arg1,
                         (returnType $arg1, "$_builder.getI64Type()")))>;
```

Explicitly-specified return types will take precedence over return types
inferred from op traits or user-defined builders. The return types of values
replacing root op results cannot be overridden.

## Debugging Tips

### Run `mlir-tblgen` to see the generated content

TableGen syntax sometimes can be obscure; reading the generated content can be a
very helpful way to understand and debug issues. To build `mlir-tblgen`, run
`cmake --build . --target mlir-tblgen` in your build directory and find the
`mlir-tblgen` binary in the `bin/` subdirectory. All the supported generators
can be found via `mlir-tblgen --help`.

To see the generated code, invoke `mlir-tblgen` with a specific generator by
providing include paths via `-I`. For example,

```sh
# To see all the C++ pattern rewrite classes
mlir-tblgen --gen-rewriters -I /path/to/mlir/include /path/to/input/td/file
```

### Compilation error: no matching member function for call to 'build'

This is because DRR is failing to call a `build()` method with result type
deduction ability. See [building operations](#building-operations) for more
details.

[TableGen]: https://llvm.org/docs/TableGen/index.html
[OpBase]: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td
