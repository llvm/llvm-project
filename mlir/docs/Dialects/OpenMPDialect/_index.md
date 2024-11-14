# 'omp' Dialect

The `omp` dialect is for representing directives, clauses and other definitions
of the [OpenMP programming model](https://www.openmp.org). This directive-based
programming model, defined for the C, C++ and Fortran programming languages,
provides abstractions to simplify the development of parallel and accelerated
programs. All versions of the OpenMP specification can be found
[here](https://www.openmp.org/specifications/).

Operations in this MLIR dialect generally correspond to a single OpenMP
directive, taking arguments that represent their supported clauses, though this
is not always the case. For a detailed information of operations, types and
other definitions in this dialect, refer to the automatically-generated
[ODS Documentation](ODS.md).

[TOC]

## Operation Naming Conventions

This section aims to standardize how dialect operation names are chosen, to
ensure a level of consistency. There are two categories of names: tablegen names
and assembly names. The former also corresponds to the C++ class that is
generated for the operation, whereas the latter is used to represent it in MLIR
text form.

Tablegen names are CamelCase, with the first letter capitalized and an "Op"
suffix, whereas assembly names are snake_case, with all lowercase letters and
words separated by underscores.

If the operation corresponds to a directive, clause or other kind of definition
in the OpenMP specification, it must use the same name split into words in the
same way. For example, the `target data` directive would become `TargetDataOp` /
`omp.target_data`, whereas `taskloop` would become `TaskloopOp` /
`omp.taskloop`.

Operations intended to carry extra information for another particular operation
or clause must be named after that other operation or clause, followed by the
name of the additional information. The assembly name must use a period to
separate both parts. For example, the operation used to define some extra
mapping information is named `MapInfoOp` / `omp.map.info`. The same rules are
followed if multiple operations are created for different variants of the same
directive, e.g. `atomic` becomes `Atomic{Read,Write,Update,Capture}Op` /
`omp.atomic.{read,write,update,capture}`.

## Clause-Based Operation Definition

One main feature of the OpenMP specification is that, even though the set of
clauses that could be applied to a given directive is independent from other
directives, these clauses can generally apply to multiple directives. Since
clauses usually define which arguments the corresponding MLIR operation takes,
it is possible (and preferred) to define OpenMP dialect operations based on the
list of clauses taken by the corresponding directive. This makes it simpler to
keep their representation consistent across operations and minimizes redundancy
in the dialect.

To achieve this, the base `OpenMP_Clause` tablegen class has been created. It is
intended to be used to create clause definitions that can be then attached to
multiple `OpenMP_Op` definitions, resulting in the latter inheriting by default
all properties defined by clauses attached, similarly to the trait mechanism.
This mechanism is implemented in
[OpenMPOpBase.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/OpenMP/OpenMPOpBase.td).

### Adding a Clause

OpenMP clause definitions are located in
[OpenMPClauses.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/OpenMP/OpenMPClauses.td).
For each clause, an `OpenMP_Clause` subclass and a definition based on it must
be created. The subclass must take a `bit` template argument for each of the
properties it can populate on associated `OpenMP_Op`s. These must be forwarded
to the base class. The definition must be an instantiation of the base class
where all these template arguments are set to `false`. The definition's name
must be `OpenMP_<Name>Clause`, whereas its base class' must be
`OpenMP_<Name>ClauseSkip`. Following this pattern makes it possible to
optionally skip the inheritance of some properties when defining operations:
[more info](#overriding-clause-inherited-properties).

Clauses can define the following properties:
  - `list<Traits> traits`: To be used when having a certain clause always
implies some op trait, like the `map` clause and the `MapClauseOwningInterface`.
  - `dag(ins) arguments`: Mandatory property holding values and attributes
used to represent the clause. Argument names use snake_case and should contain
the clause name to avoid name clashes between clauses. Variadic arguments
(non-attributes) must contain the "_vars" suffix.
  - `string {req,opt}AssemblyFormat`: Optional formatting strings to produce
custom human-friendly printers and parsers for arguments associated with the
clause. It will be combined with assembly formats for other clauses as explained
[below](#adding-an-operation).
  - `string description`: Optional description text to describe the clause and
its representation.
  - `string extraClassDeclaration`: Optional C++ declarations to be added to
operation classes including the clause.

For example:

```tablegen
class OpenMP_ExampleClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<AnyType>:$example_var
  );

  let optAssemblyFormat = [{
    `example` `(` $example_var `:` type($example_var) `)`
  }];

  let description = [{
    The `example_var` argument defines the variable to which the EXAMPLE clause
    applies.
  }];
}

def OpenMP_ExampleClause : OpenMP_ExampleClauseSkip<>;
```

### Adding an Operation

Operations in the OpenMP dialect, located in
[OpenMPOps.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td),
can be defined like any other regular operation by just specifying a `mnemonic`
and optional list of `traits` when inheriting from `OpenMP_Op`, and then
defining the expected `description`, `arguments`, etc. properties inside of its
body. However, in most cases, basing the operation definition on its list of
accepted clauses is significantly simpler because some of the properties can
just be inherited from these clauses.

In general, the way to achieve this is to specify, in addition to the `mnemonic`
and optional list of `traits`, a list of `clauses` where all the applicable
`OpenMP_<Name>Clause` definitions are added. Then, the only properties that
would have to be defined in the operation's body are the `summary` and
`description`. For the latter, only the operation itself would have to be
defined, and the description for its clause-inherited arguments is appended
through the inherited `clausesDescription` property. By convention, the list of
clauses for an operation must be specified in alphabetical order.

If the operation is intended to have a single region, this is better achieved by
setting the `singleRegion=true` template argument of `OpenMP_Op` rather manually
populating the `regions` property of the operation, because that way the default
`assemblyFormat` is also updated correspondingly.

For example:

```tablegen
def ExampleOp : OpenMP_Op<"example", traits = [
    AttrSizedOperandSegments, ...
  ], clauses = [
    OpenMP_AlignedClause, OpenMP_IfClause, OpenMP_LinearClause, ...
  ], singleRegion = true> {
  let summary = "example construct";
  let description = [{
    The example construct represents...
  }] # clausesDescription;
}
```

This is possible because the `arguments`, `assemblyFormat` and
`extraClassDeclaration` properties of the operation are by default
populated by concatenating the corresponding properties of the clauses on the
list. In the case of the `assemblyFormat`, this involves combining the
`reqAssemblyFormat` and the `optAssemblyFormat` properties. The
`reqAssemblyFormat` of all clauses is concatenated first and separated using
spaces, whereas the `optAssemblyFormat` is wrapped in an `oilist()` and
interleaved with "|" instead of spaces. The resulting `assemblyFormat` contains
the required assembly format strings, followed by the optional assembly format
strings, optionally the `$region` and the `attr-dict`.

### Overriding Clause-Inherited Properties

Although the clause-based definition of operations can greatly reduce work, it's
also somewhat restrictive, since there may be some situations where only part of
the operation definition can be automated in that manner. For a fine-grained
control over properties inherited from each clause two features are available:

  - Inhibition of properties. By using `OpenMP_<Name>ClauseSkip` tablegen
classes, the list of properties copied from the clause to the operation can be
selected. For example, `OpenMP_IfClauseSkip<assemblyFormat = true>` would result
in every property defined for the `OpenMP_IfClause` except for the
`assemblyFormat` being used to initially populate the properties of the
operation.
  - Augmentation of properties. There are times when there is a need to add to
a clause-populated operation property. Instead of overriding the property in the
definition of the operation and having to manually replicate what would
otherwise be automatically populated before adding to it, some internal
properties are defined to hold this default value: `clausesArgs`,
`clausesAssemblyFormat`, `clauses{Req,Opt}AssemblyFormat` and
`clausesExtraClassDeclaration`.

In the following example, assuming both the `OpenMP_InReductionClause` and the
`OpenMP_ReductionClause` define a `getReductionVars` extra class declaration,
we skip the conflicting `extraClassDeclaration`s inherited by both clauses and
provide another implementation, without having to also re-define other
declarations inherited from the `OpenMP_AllocateClause`:

```tablegen
def ExampleOp : OpenMP_Op<"example", traits = [
    AttrSizedOperandSegments, ...
  ], clauses = [
    OpenMP_AllocateClause,
    OpenMP_InReductionClauseSkip<extraClassDeclaration = true>,
    OpenMP_ReductionClauseSkip<extraClassDeclaration = true>
  ], singleRegion = true> {
  let summary = "example construct";
  let description = [{
    This operation represents...
  }] # clausesDescription;

  // Override the clause-populated extraClassDeclaration and add the default
  // back via appending clausesExtraClassDeclaration to it. This has the effect
  // of adding one declaration. Since this property is skipped for the
  // InReduction and Reduction clauses, clausesExtraClassDeclaration won't
  // incorporate the definition of this property for these clauses.
  let extraClassDeclaration = [{
    SmallVector<Value> getReductionVars() {
      // Concatenate inReductionVars and reductionVars and return the result...
    }
  }] # clausesExtraClassDeclaration;
}
```

These features are intended for complex edge cases, but an effort should be made
to avoid having to use them, since they may introduce inconsistencies and
complexity to the dialect.

### Tablegen Verification Pass

As a result of the implicit way in which fundamental properties of MLIR
operations are populated following this approach, and the ability to override
them, forgetting to append clause-inherited values might result in hard to debug
tablegen errors.

For this reason, the `-verify-openmp-ops` tablegen pseudo-backend was created.
It runs before any other tablegen backends are triggered for the
[OpenMPOps.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td)
file and warns any time a property defined for a clause is not found in the
corresponding operation, except if it is explicitly skipped as described
[above](#overriding-clause-inherited-properties). This way, in case of a later
tablegen failure while processing OpenMP dialect operations, earlier messages
triggered by that pass can point to a likely solution.

### Operand Structures

One consequence of basing the representation of operations on the set of values
and attributes defined for each clause applicable to the corresponding OpenMP
directive is that operation argument lists tend to be long. This has the effect
of making C++ operation builders difficult to work with and easy to mistakenly
pass arguments in the wrong order, which may sometimes introduce hard to detect
problems.

A solution provided to this issue are operand structures. The main idea behind
them is that there is one defined for each clause, holding a set of fields that
contain the data needed to initialize each of the arguments associated with that
clause. Clause operand structures are aggregated into operation operand
structures via class inheritance. Then, a custom builder is defined for each
operation taking the corresponding operand structure as a parameter. Since each
argument is a named member of the structure, it becomes much simpler to set up
the desired arguments to create a new operation.

Ad-hoc operand structures available for use within the ODS definition of custom
operation builders might be defined in
[OpenMPClauseOperands.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/OpenMP/OpenMPClauseOperands.h).
However, this is generally not needed for clause-based operation definitions.
The `-gen-openmp-clause-ops` tablegen backend, triggered when building the 'omp'
dialect, will automatically produce structures in the following way:

- It will create a `<Name>ClauseOps` structure for each `OpenMP_Clause`
definition with one field per argument.
- The name of each field will match the tablegen name of the corresponding
argument, except for replacing snake case with camel case.
- The type of the field will be obtained from the corresponding tablegen
argument's type:
  - Values are represented with `mlir::Value`, except for `Variadic`, which
  makes it an `llvm::SmallVector<mlir::Value>`.
  - `OptionalAttr` is represented by the translation of its `baseAttr`.
  - `TypedArrayAttrBase`-based attribute types are represented by wrapping the
  translation of their `elementAttr` in an `llvm::SmallVector`. The only
  exception for this case is if the `elementAttr` is a "scalar" (i.e. non
  array-like) attribute type, in which case the more generic `mlir::Attribute`
  will be used in place of its `storageType`.
  - For `ElementsAttrBase`-based attribute types a best effort is attempted to
  obtain an element type (`llvm::APInt`, `llvm::APFloat` or
  `DenseArrayAttrBase`'s `returnType`) to be wrapped in an `llvm::SmallVector`.
  If it cannot be obtained, which will happen with non-builtin direct subclasses
  of `ElementsAttrBase`, a warning will be emitted and the `storageType` (i.e.
  specific `mlir::Attribute` subclass) will be used instead.
  - Other attribute types will be represented with their `storageType`.
- It will create `<Name>Operands` structure for each operation, which is an
empty structure subclassing all operand structures defined for the corresponding
`OpenMP_Op`'s clauses.

### Entry Block Argument-Defining Clauses

In their MLIR representation, certain OpenMP clauses introduce a mapping between
values defined outside the operation they are applied to and entry block
arguments for the region of that MLIR operation. This enables, for example, the
introduction of private copies of the same underlying variable defined outside
the MLIR operation the clause is attached to. Currently, clauses with this
property can be classified into three main categories:
  - Map-like clauses: `map`, `use_device_addr` and `use_device_ptr`.
  - Reduction-like clauses: `in_reduction`, `reduction` and `task_reduction`.
  - Privatization clauses: `private`.

All three kinds of entry block argument-defining clauses use a similar custom
assembly format representation, only differing based on the different pieces of
information attached to each kind. Below, one example of each is shown:

```mlir
omp.target map_entries(%x -> %x.m, %y -> %y.m : !llvm.ptr, !llvm.ptr) {
  // Use %x.m, %y.m in place of %x and %y...
}

omp.wsloop reduction(@add.i32 %x -> %x.r, byref @add.f32 %y -> %y.r : !llvm.ptr, !llvm.ptr) {
  // Use %x.r, %y.r in place of %x and %y...
}

omp.parallel private(@x.privatizer %x -> %x.p, @y.privatizer %y -> %y.p : !llvm.ptr, !llvm.ptr) {
  // Use %x.p, %y.p in place of %x and %y...
}
```

As a consequence of parsing and printing the operation's first region entry
block argument names together with the custom assembly format of these clauses,
entry block arguments (i.e. the `^bb0(...):` line) must not be explicitly
defined for these operations. Additionally, it is not possible to implement this
feature while allowing each clause to be independently parsed and printed,
because they need to be printed/parsed together with the corresponding
operation's first region. They must have a well-defined ordering in which
multiple of these clauses are specified for a given operation, as well.

The parsing/printing of these clauses together with the region provides the
ability to define entry block arguments directly after the `->`. Forcing a
specific ordering between these clauses makes the block argument ordering
well-defined, which is the property used to easily match each clause with the
entry block arguments defined by it.

Custom printers and parsers for operation regions based on the entry block
argument-defining clauses they take are implemented based on the
`{parse,print}BlockArgRegion` functions, which take care of the sorting and
formatting of each kind of clause, minimizing code duplication resulting from
this approach. One example of the custom assembly format of an operation taking
the `private` and `reduction` clauses is the following:

```tablegen
let assemblyFormat = clausesAssemblyFormat # [{
  custom<PrivateReductionRegion>($region, $private_vars, type($private_vars),
      $private_syms, $reduction_vars, type($reduction_vars), $reduction_byref,
      $reduction_syms) attr-dict
}];
```

The `BlockArgOpenMPOpInterface` has been introduced to simplify the addition and
handling of these kinds of clauses. It holds `num<ClauseName>BlockArgs()`
functions that by default return 0, to be overriden by each clause through the
`extraClassDeclaration` property. Based on these functions and the expected
alphabetical sorting between entry block argument-defining clauses, it
implements `get<ClauseName>BlockArgs()` functions that are the intended method
of accessing clause-defined block arguments.

## Loop-Associated Directives

Loop-associated OpenMP constructs are represented in the dialect as loop wrapper
operations. These implement the `LoopWrapperInterface`, which enforces a series
of restrictions upon the operation:
  - It has the `NoTerminator` and `SingleBlock` traits;
  - It contains a single region; and
  - Its only block contains exactly one operation, which must be another loop
wrapper or `omp.loop_nest` operation.

This approach splits the representation for a loop nest and the loop-associated
constructs that specify how its iterations are executed, possibly across various
SIMD lanes (`omp.simd`), threads (`omp.wsloop`), teams of threads
(`omp.distribute`) or tasks (`omp.taskloop`). The ability to directly nest
multiple loop wrappers to impact the execution of a single loop nest is used to
represent composite constructs in a modular way.

The `omp.loop_nest` operation represents a collapsed rectangular loop nest that
must always be wrapped by at least one loop wrapper, which defines how it is
intended to be executed. It serves as a simpler and more restrictive
representation of OpenMP loops while a more general approach to support
non-rectangular loop nests, loop transformations and non-perfectly nested loops
based on a new `omp.canonical_loop` definition is developed.

The following example shows how a `parallel {do,for}` construct would be
represented:
```mlir
omp.parallel ... {
  ...
  omp.wsloop ... {
    omp.loop_nest (%i) : index = (%lb) to (%ub) step (%step) {
      %a = load %a[%i] : memref<?xf32>
      %b = load %b[%i] : memref<?xf32>
      %sum = arith.addf %a, %b : f32
      store %sum, %c[%i] : memref<?xf32>
      omp.yield
    }
  }
  ...
  omp.terminator
}
```

### Loop Transformations

In addition to the worksharing loop-associated constructs described above, the
OpenMP specification also defines a set of loop transformation constructs. They
replace the associated loop(s) before worksharing constructs are executed on the
generated loop(s). Some examples of such constructs are `tile` and `unroll`.

A general approach for representing these types of OpenMP constructs has not yet
been implemented, but it is closely linked to the `omp.canonical_loop` work.
Nevertheless, loop transformation that the `collapse` clause for loop-associated
worksharing constructs defines can be represented by introducing multiple
bounds, step and induction variables to the `omp.loop_nest` operation.

## Compound Construct Representation

The OpenMP specification defines certain shortcuts that allow specifying
multiple constructs in a single directive, which are referred to as compound
constructs (e.g. `parallel do` contains the `parallel` and `do` constructs).
These can be further classified into [combined](#combined-constructs) and
[composite](#composite-constructs) constructs. This section describes how they
are represented in the dialect.

When clauses are specified for compound constructs, the OpenMP specification
defines a set of rules to decide to which leaf constructs they apply, as well as
potentially introducing some other implicit clauses. These rules must be taken
into account by those creating the MLIR representation, since it is a per-leaf
representation that expects these rules to have already been followed.

### Combined Constructs

Combined constructs are semantically equivalent to specifying one construct
immediately nested inside another. This property is used to simplify the dialect
by representing them through the operations associated to each leaf construct.
For example, `target teams` would be represented as follows:

```mlir
omp.target ... {
  ...
  omp.teams ... {
    ...
    omp.terminator
  }
  ...
  omp.terminator
}
```

### Composite Constructs

Composite constructs are similar to combined constructs in that they specify the
effect of one construct being applied immediately after another. However, they
group together constructs that cannot be directly nested into each other.
Specifically, they group together multiple loop-associated constructs that apply
to the same collapsed loop nest.

As of version 5.2 of the OpenMP specification, the list of composite constructs
is the following:
  - `{do,for} simd`;
  - `distribute simd`;
  - `distribute parallel {do,for}`;
  - `distribute parallel {do,for} simd`; and
  - `taskloop simd`.

Even though the list of composite constructs is relatively short and it would
also be possible to create dialect operations for each, it was decided to
allow attaching multiple loop wrappers to a single loop instead. This minimizes
redundancy in the dialect and maximizes its modularity, since there is a single
operation for each leaf construct regardless of whether it can be part of a
composite construct. On the other hand, this means the `omp.loop_nest` operation
will have to be interpreted differently depending on how many and which loop
wrappers are attached to it.

To simplify the detection of operations taking part in the representation of a
composite construct, the `ComposableOpInterface` was introduced. Its purpose is
to handle the `omp.composite` discardable dialect attribute that can optionally
be attached to these operations. Operation verifiers will ensure its presence is
consistent with the context the operation appears in, so that it is valid when
the attribute is present if and only if it represents a leaf of a composite
construct.

For example, the `distribute simd` composite construct is represented as
follows:

```mlir
omp.distribute ... {
  omp.simd ... {
    omp.loop_nest (%i) : index = (%lb) to (%ub) step (%step) {
      ...
      omp.yield
    }
  } {omp.composite}
} {omp.composite}
```

One exception to this is the representation of the
`distribute parallel {do,for}` composite construct. The presence of a
block-associated `parallel` leaf construct would introduce many problems if it
was allowed to work as a loop wrapper. In this case, the "hoisted `omp.parallel`
representation" is used instead. This consists in making `omp.parallel` the
parent operation, with a nested `omp.loop_nest` wrapped by `omp.distribute` and
`omp.wsloop` (and `omp.simd`, in the `distribute parallel {do,for} simd` case).

This approach works because `parallel` is a parallelism-generating construct,
whereas `distribute` is a worksharing construct impacting the higher level
`teams` construct, making the ordering between these constructs not cause
semantic mismatches. This property is also exploited by LLVM's SPMD-mode.

```mlir
omp.parallel ... {
  ...
  omp.distribute ... {
    omp.wsloop ... {
      omp.loop_nest (%i) : index = (%lb) to (%ub) step (%step) {
        ...
        omp.yield
      }
    } {omp.composite}
  } {omp.composite}
  ...
  omp.terminator
} {omp.composite}
```
