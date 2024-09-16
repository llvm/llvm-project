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
through the inherited `clausesDescription` property.

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

## Loop-Associated Directives

Loop-associated OpenMP constructs are represented in the dialect as loop wrapper
operations. These implement the `LoopWrapperInterface`, which enforces a series
of restrictions upon the operation:
  - It contains a single region with a single block; and
  - Its block contains exactly two operations: another loop wrapper or
`omp.loop_nest` operation and a terminator.

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
    omp.terminator
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
