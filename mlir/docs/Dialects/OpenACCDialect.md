# 'acc' Dialect

The `acc` dialect is an MLIR dialect for representing the OpenACC
programming model. OpenACC is a standardized directive-based model which
is used with C, C++, and Fortran to enable programmers to expose
parallelism in their code. The descriptive approach used by OpenACC
allows targeting of parallel multicore and accelerator targets like GPUs
by giving the compiler the freedom of how to parallelize for specific
architectures. OpenACC also provides the ability to optimize the
parallelism through increasingly more prescriptive clauses.

This dialect models the constructs from the 
[OpenACC 3.3 specification](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.3-final.pdf)

This document describes the design of the OpenACC dialect in MLIR. It
lists and explains design goals and design choices along with their
rationale. It also describes specifics with regards to acc dialect
operations, types, and attributes.

[TOC]

## Dialect Design Goals

* Needs to have complete representation of the OpenACC language.
	- A frontend requires this in order to properly generate a
	representation of possible `acc` pragmas in MLIR. Additionally,
	this dialect is expected to be further lowered when materializing
	its semantics. Without a complete representation, a frontend might
	choose a lower abstraction (such as direct runtime call) - but this
	would impact the ability to do analysis and optimizations on the
	dialect.
* Allow representation at the same semantic level as the OpenACC
language while having capability to represent nuances of the source
language semantics (such as Fortran descriptors) in an agnostic manner.
	- Using abstractions that closely model the OpenACC language
	simplifies frontend implementation. It also allows for easier
	debugging of the IR. However, sometimes source language specific
	behavior is needed when materializing OpenACC. In these cases, such
	as privatization of C++ objects with default constructor, the
	frontend fills in the `recipe` along with the `private` operation
	which can be packaged neatly with the `acc` dialect operations.
* Be able to regenerate the semantic equivalent of the user pragmas from
the dialect (including bounds, names, clauses, modifiers, etc).
	- This is a strong measure of making sure that the dialect is not
	lossy in semantics. It also allows capability to generate
	appropriate and useful debug information outside of the frontend.
* Be dialect agnostic so that it can be used and coexist with other
dialects including but not limited to `hlfir`, `fir`, `llvm`, `cir`.
	- Directive-based models such as OpenACC are always used with a
	source language, so the `acc` dialect coexisting with other
	dialect(s) is necessary by construction. Through proper
	abstractions, neither the `acc` dialect nor the source language
	dialect should have dependencies on each other; where needed,
	interfaces should be used to ensure `acc` dialect can verify
	expected properties.
* The dialect must allow dataflow to be modeled accurately and
performantly using MLIR's existing facilities.
	- Appropriate dataflow modeling is important for analyses and IR
	reasoning - even something as simple as walking the uses. Therefore
	operations, like data operations, are expected to generate results
	which can be used in modeling behavior. For example, consider an
	`acc copyin` clause. After the `acc.copyin` operation, a pointer
	which lives on devices should be distinguishable from one that lives
	in host memory.
* Be friendly to MLIR optimization passes by implementing common
interfaces.
	- Interfaces, such as `MemoryEffects`, are the key way MLIR
	transformations and analyses are designed to interact with the IR.
	In order for the operations in the `acc` dialect to be optimizable
	(either directly or even indirectly by not blocking optimizations
	of nested IR), implementing relevant common interfaces is needed.

The design philosophy of the acc dialect is one where the design goals
are adhered to. Current and planned operations, attributes, types must
adhere to the design goals.

## Operation Categories

The OpenACC dialect includes both high-level operations (which retain
the same semantic meaning as their OpenACC language equivalent),
intermediate-level operations (which are used to decompose clauses
from constructs), and low-level operations (to encode specifics
associated with source language in a generic way).

The high-level operations list contains the following OpenACC language
constructs and their corresponding operations:
* `acc parallel` &rarr; `acc.parallel`
* `acc kernels` &rarr; `acc.kernels`
* `acc serial` &rarr; `acc.serial`
* `acc data` &rarr; `acc.data`
* `acc loop` &rarr; `acc.loop`
* `acc enter data` &rarr; `acc.enter_data`
* `acc exit data` &rarr; `acc.exit_data`
* `acc host_data` &rarr; `acc.host_data`
* `acc init` &rarr; `acc.init`
* `acc shutdown` &rarr; `acc.shutdown`
* `acc update` &rarr; `acc.update`
* `acc set` &rarr; `acc.set`
* `acc wait` &rarr; `acc.wait`
* `acc atomic read` &rarr; `acc.atomic.read`
* `acc atomic write` &rarr; `acc.atomic.write`
* `acc atomic update` &rarr; `acc.atomic.update`
* `acc atomic capture` &rarr; `acc.atomic.capture`

This second group contains operations which are used to represent
either decomposed constructs or clauses for more accurate modeling:
* `acc routine` &rarr; `acc.routine` + `acc.routine_info` attribute
* `acc declare` &rarr; `acc.declare_enter` + `acc.declare_exit` or
`acc.declare`
* `acc {construct} copyin` &rarr; `acc.copyin` (before region) +
`acc.delete` (after region)
* `acc {construct} copy` &rarr; `acc.copyin` (before region) +
`acc.copyout` (after region)
* `acc {construct} copyout` &rarr; `acc.create` (before region) +
`acc.copyout` (after region)
* `acc {construct} attach` &rarr; `acc.attach` (before region) +
`acc.detach` (after region)
* `acc {construct} create` &rarr; `acc.create` (before region) +
`acc.delete` (after region)
* `acc {construct} present` &rarr; `acc.present` (before region) +
`acc.delete` (after region)
* `acc {construct} no_create` &rarr; `acc.nocreate` (before region) +
`acc.delete` (after region)
* `acc {construct} deviceptr` &rarr; `acc.deviceptr`
* `acc {construct} private` &rarr; `acc.private`
* `acc {construct} firstprivate` &rarr; `acc.firstprivate`
* `acc {construct} reduction` &rarr; `acc.reduction`
* `acc cache` &rarr; `acc.cache`
* `acc update device` &rarr; `acc.update_device`
* `acc update host` &rarr; `acc.update_host`
* `acc host_data use_device` &rarr; `acc.use_device`
* `acc declare device_resident` &rarr; `acc.declare_device_resident`
* `acc declare link` &rarr; `acc.declare_link`
* `acc exit data delete` &rarr; `acc.delete` (with `structured` flag as
false)
* `acc exit data detach` &rarr; `acc.detach` (with `structured` flag as
false)
* `acc {construct} {data_clause}(var[lb:ub])` &rarr; `acc.bounds`

The low-level operations are:
* `acc.private.recipe`
* `acc.reduction.recipe`
* `acc.firstprivate.recipe`
* `acc.global_ctor`
* `acc.global_dtor`
* `acc.yield`
* `acc.terminator`
The low-level operations semantics and reasoning are further explained
in sections below.

### Data Operations

#### Data Clause Decomposition
The data clauses are decomposed from their constructs for better
dataflow modeling in MLIR. There are multiple reasons for this which
are consistent with the dialect goals:
* Correctly represents dataflow. Data clauses have different effects
at entry to region and at exit from region.
* Friendlier to add attributes such as `MemoryEffects` to a single
operation. This can better reflect semantics (like the fact that an
`acc.copyin` operation only reads host memory)
* Operations can be moved or optimized individually (eg `CSE`).
* Easier to keep track of debug information. Line location can point to
the text representing the data clause instead of the construct.
Additionally, attributes can be used to keep track of variable names in
clauses without having to walk the IR tree in attempt to recover the
information (this makes acc dialect more agnostic with regards to what
other dialect it is used with).
* Clear operation ordering since all data operations are on same
list.

Each of the `acc` dialect data operations represents either the
entry or the exit portion of the data action specification. Thus,
`acc.copyin` represents the semantics defined in section
`2.7.7 copyin clause` whose wording starts with
`At entry to a region`. The decomposed exit operation `acc.delete`
represents the second part of that section, whose wording starts with
`At exit from the region`. The `delete` action may be performed
after checking and updating of the relevant reference counters noted.

The `acc` data operations, even when decomposed, retain their original
data clause in an operation operand `dataClause` for possibility to
recover this information during debugging. For example, `acc copy`,
does not translate to `acc.copy` operation, but instead to `acc.copyin`
for entry and `acc.copyout` for exit. Both the decomposed operations
hold a `dataClause` field that specifies this was an `acc copy`.

The link between the decomposed entry and exit operations is the ssa
value produced by the entry operation. Namely, it is the `accPtr` result
which is used both in the `dataOperands` of the operation used for the
construct and in the `accPtr` operand of the exit operation.

#### Bounds

OpenACC data clauses allow the use of bounds specifiers as per
`2.7.1 Data Specification in Data Clauses`. However, array dimensions
for the data are not always required in the clause if the source
language's type system captures this information - the user can just
specify the variable name in the data clause. So the `acc.bounds`
operation is an important piece to ensure uniform representation of both
explicit user set dimensions and implicit type-based dimensions. It
contains several key features to allow properly encoding sizes in a
manner flexible and agnostic to the source language's dialect:
* Multi-dimensional arrays can be represented by using multiple ordered
`acc.bounds` operations.
* Bounds are required to be zero-normalized. This works well with the
`PointerLikeType` requirement in data clauses - since a lowerbound of 0
means looking at data at the zero offset from pointer. This requirement
also works well in ensuring the `acc` dialect is agnostic to source
language dialect since it prevents ambiguity such as the case of Fortran
arrays where the lower bound is not a fixed value.
* If the source dialect does not encode the dimensions in the type (eg
`!fir.array<?x?xi32>`) but instead encodes it in some other way (such as
through descriptors), then the frontend must fill in the `acc.bounds`
operands with appropriate information (such as loads from descriptor).
The `acc.bounds` operation also permits lossy source dialect, such
as if the frontend uses aggressive pointer decay and cannot represent
the dimensions in the type system (eg using `!llvm.ptr` for arrays).
Both of these aspects show `acc.bounds`' operation's flexibility to
allow the representation to be agnostic since the `acc` dialect is not
expected to be able to understand how to extract dimension information
from the types of the source dialect.
* The OpenACC specification allows either extent or upperbound in the
data clause depending on whether it is Fortran or C and C++. The
`acc.bounds` operation is rich enough to accept either or both - for
convenience in lowering to the dialect and for ability to precisely
capture the meaning from the clause.
* The stride, either in units or bytes, can be also captured in the
`acc.bounds` operation. This is also an important part to be able to
accept a source language's arrays without forcing the frontend to
normalize them in some way. For example, consider a case where in a
parent function, a whole array is mapped to device. Then only a view of
a non-1 stride is passed to child function (eg Fortran array slice with
non-1 stride). A `copy` operation of this data in child should be able
to avoid remapping this array. If instead the operation required
normalizing the array (such as making it contiguous), then unexpected
disjoint mapping of the same host data would be error-prone since it
would result in multiple mappings to device.

#### Counters

The data operations also maintain semantics described in the OpenACC
specification related to runtime counters. More specifically, consider
the specification of the entry portion of `acc copyin` in section 2.7.7:
```
At entry to a region, the structured reference counter is used. On an
enter data directive, the dynamic reference counter is used.
- If var is present and is not a null pointer, a present increment
action with the appropriate reference counter is performed.
- If var is not present, a copyin action with the appropriate reference
counter is performed.
- If var is a pointer reference, an attach action is performed.
```
The `acc.copyin` operation includes these semantics, including those
related to attach, which is specified through the `varPtrPtr` operand.
The `structured` flag on the operation is important since the
`structured reference counter` should be used when the flag is true; and
the `dynamic reference counter` should be used when it is false.

At exit from structured regions (`acc data`, `acc kernels`), the
`acc copyin` operation is decomposed to `acc.delete` (with the
`structured` flag as true). The semantics of the `acc.delete` are
also consistent with the OpenACC specification noted for the exit
portion of the `acc copyin` clause:
```
At exit from the region:
- If the structured reference counter for var is zero, no action is
taken.
- Otherwise, a detach action is performed if var is a pointer reference,
and a present decrement action with the structured reference counter is
performed if var is not a null pointer. If both structured and dynamic
reference counters are zero, a delete action is performed.
```

### Types

Since the `acc dialect` is meant to be used alongside other dialects which
represent the source language, appropriate use of types and type interfaces is
key to ensuring compatibility. This section describes those considerations.

#### Data Clause Operation Types

Data clause operations (eg. `acc.copyin`) rely on the following type
considerations:
* type of acc data clause operation input `var`
	- The type of `var` must be one with `PointerLikeType` or `MappableType`
	interfaces attached. The first, `PointerLikeType`, is useful because
	the OpenACC memory model distinguishes between host and device memory
	explicitly - and the mapping between the two is	done through pointers. Thus,
	by explicitly requiring it in the dialect, the appropriate language
	frontend must create storage or	use type that satisfies the mapping
	constraint. The second possibility, `MappableType` was added because
	memory/storage concept is a lower level abstraction and not all dialects
	choose to use a pointer abstraction especially in the case where semantics
	are more complex (such as `fir.box` which represents Fortran descriptors
	and is defined in the `fir` dialect used from `flang`).
* type of result of acc data clause operations
	- The type of the acc data clause operation is exactly the same as
	`var`. This was done intentionally instead of introducing specific `acc`
	output types so that so that IR compatibility and the dialect's
	existing strong type checking can be maintained. This is needed
	since the `acc` dialect must live within another dialect whose type
	system is unknown to it.
* variable type captured in `varType`
	- When `var`'s type is `PointerLikeType`, the actual type of the target
	may be lost. More specifically, dialects like `llvm` which use opaque
	pointers, do not record the target variable's type. The use of this field
	bridges this gap.
* type of decomposed clauses
	- Decomposed clauses, such as `acc.bounds` and `acc.declare_enter`
	produce types to allow their results to be used only in specific
	operations. These are synthetic types solely used for proper IR
	construction.

#### Pointer-Like Requirement

The need to have pointer-type requirement in the acc dialect stems from
a few different aspects:
- Existing dialects like `hlfir`, `fir`, `cir`, `llvm` use a pointer
representation for variables.
- Reference counters (for data clauses) are described in terms of
memory. In OpenACC spec 3.3 in section 2.6.7. It says: "A structured reference
counter is incremented when entering each data or compute region that contain an
explicit data clause or implicitly-determined data attributes for that section
of memory". This implies addressability of memory.
- Attach semantics (2.6.8 attachment counter) are specified using
"address" terminology: "The attachment counter for a pointer is set to
one whenever the pointer is attached to new target address, and
incremented whenever an attach action for that pointer is performed for
the same target address.

#### Type Interfaces

The `acc` dialect describes two different type interfaces which must be
implemented and attached to the source dialect's types in order to allow use
of data clause operations (eg. `acc.copyin`). They are as follows:
* `PointerLikeType`
  - The idea behind this interface is that variables end up being represented
  as pointers in many dialects. More specifically, `fir`, `cir`, `llvm`
  represent user declared local variables with some dialect specific form of
  `alloca` operation which produce pointers. Globals, similarly, are referred by
  their address through some form of `address_of` operation. Additionally, an
  implementation for OpenACC runtime needs to distinguish between device and
  host memory - also typically done by talking about pointers. So this type
  interface requirement fits in naturally with OpenACC specification. Data
  mapping operation semantics can often be simply described by a pointer and
  size of the data it points to.
* `MappableType`
   - This interface was introduced because the `PointerLikeType` requirement
  cannot represent cases when the source dialect does not use pointers. Also,
  some cases, such as Fortran descriptor-backed arrays and Fortran optional
  arguments, require decomposition into multiple steps. For example, in the
  descriptor case, mapping of descriptor is needed, mapping of the data, and
  implicit attach into device descriptor. In order to allow capturing all of
  this complexity with a single data clause operation, the `MappableType`
  interface was introduced. This is consistent with the dialect's goals
  including being "able to regenerate the semantic equivalent of the user
  pragmas".

The intent is that a dialect's type system implements one of these two
interfaces. And to be precise, a type should only implement one or the other
(and not both) - since keeping them separate avoids ambiguity on what actually
needs mapped. When `var` is `PointerLikeType`, the assumption is that the data
pointed-to will be mapped. If the pointer-like type also implemented
`MappableType` interface, it becomes ambiguous whether the data pointed to or
the pointer itself is being mapped.

### Recipes

Recipes are a generic way to express source language specific semantics.

There are currently two categories of recipes, but the recipe concept
can be extended for any additional low-level information that needs
to be captured for successful lowering of OpenACC. The two categories
are:
* recipes used in the context of privatization associated with a
construct
* recipes used in the context of additional specification of data
semantics

The intention of the recipes is to specify how materialization of
action, such as privatization, should be done when the semantics
of the action needs interpreted and lowered, such as before generating
LLVM dialect.

The recipes used for privatization provide a source-language independent
way of specifying the creation of a local variable of that type. This
means using the appropriate `alloca` instruction and being able to
specify default initialization or default constructor.

### Routine

The routine directive is used to note that a procedure should be made
available for the accelerator in a way that is consistent with its
modifiers, such as those that describe the parallelism. In the acc
dialect, an acc routine is represented through two joint pieces - an
attribute and an operation:
* The `acc.routine` operation is simply a specifier which notes which
symbol (or string) the acc routine is needed for, along with parallelism
associated. This defines a symbol that can be referenced in attribute.
* The `acc.routine_info` attribute is an attribute used on the source
dialect specific operation which specifies one or multiple `acc.routine`
symbols. Typically, this is attached to `func.func` which either 
provides the declaration (in case of externals) or provides the
actual body of the acc routine in the dialect that the source language
was translated to.

### Declare

OpenACC `declare` is a mechanism which declares a definition of a global
or a local to be accessible to accelerator with an implicit lifetime
as that of the scope where it was declared in. Thus, `declare` semantics
are represented through multiple operations and attributes:
* `acc.declare` - This is a structured operation which contains an
MLIR region and can be used in similar manner as acc.data to specify
an implicit data region with specific procedure lifetime. This is
typically used inside `func.func` after variable declarations.
* `acc.declare_enter` - This is an unstructured operation which is
used as a decomposed form of `acc declare`. It effectively allows the
entry operation to exist in a scope different than the exit operation.
It can also be used along `acc.declare_exit` which consumes its token
to define a scoped region without using MLIR region. This operation is
also used in `acc.global_ctor`.
* `acc.declare_exit` - The matching equivalent of `acc.declare_enter`
except that it specifies exit semantics. This operation is typically
used inside a `func.func` at the exit points or with `acc.global_dtor`.
* `acc.global_ctor` - Lives at the same level as source dialect globals
and is used to specify data actions to be done at program entry. This
is used in conjunction with source dialect globals whose lifetime is
not just a single procedure.
* `acc.global_dtor` - Defines the exit data actions that should be done
at program exit. Typically used to revert the actions of
`acc.global_ctor`.

The attributes:
* `acc.declare` - This is a facility for easier determination of
variables which are `acc declare`'d. This attribute is used on
operations producing globals and on operations producing locals such as
dialect specific `alloca`'s. Having this attribute is required in order
to appear in a data mapping operation associated with any of the
`acc.declare*` operations.
* `acc.declare_action` - Since the OpenACC specification allows
declaration of variables that have yet to be allocated, this attribute
is used at the allocation and deallocation points. More specifically,
this attribute captures symbols of functions to be called to perform
an action either pre-allocate, post-allocate, pre-deallocate, or
post-deallocate. Calls to these functions should be materialized when
lowering OpenACC semantics to ensure proper data actions are done
after the allocation/deallocation.

## OpenACC Transforms and Analyses

The design goal for the `acc` dialect is to be friendly to MLIR
optimization passes including CSE and LICM. Additionally, since it is
designed to recover original clauses, it makes late verification and
analysis possible in the MLIR framework outside of the frontend.

This section describes a few MLIR-level passes for which the `acc`
dialect design should be friendly for. This section is currently
solely outlining the possibilities intended by the design and not
necessarily existing passes.

### Verification

Since the OpenACC dialect is not lossy with regards to its
representation, it is possible to do OpenACC language semantic checking
at the MLIR-level. What follows is a list of various semantic checks
needed.

This first list is required to be done in the frontend because the `acc`
dialect operations must be valid when constructed:
* Ensure that only listed clauses are allowed for each directive.
* Ensure that only listed modifiers are allowed for each clause.

However, the following are semantic checks that can be done at the
MLIR-level (either in a separate pass or as part of the operation
verifier):
* Specify the validity checks that each modifier needs. (eg num_gangs
may need a positive integer).
* Ensure valid clause nesting.
* Validate clause restrictions which cannot appear with others.
* Validate that no conflicting clauses are used on variables.

Note that some of these checks can be even more precise when done at the
MLIR level because optimizations like inlining and constant propagation
expose detail that wouldn't have been visible in the frontend.

### Implicit Data Attributes

The OpenACC specification includes a section on `2.6.2 Variables with
Implicitly Determined Data Attributes`. What this section describes are
the data actions that should be applied to a variable for which
user did not specify a data action for. The action depends on the
construct being used and also on the default clause. However, the point
to note here is that variables which are live-in into the acc region
must employ some data mapping so the data can be passed to accelerator.

One possible optimizations that affects data attributes needed is
`Scalar Replacement of Aggregates (SROA)`. The `acc` dialect should
not prevent this from happening on the source dialect.

Because it is intended to be possible to apply optimizations across an
`acc` region, the analysis/transformation pass that applies the implicit
data attributes should be run as late as possible - ideally right before
any outlining process which uses the `acc` region body to create an
accelerator procedure. It is expected that existing MLIR facilities,
such as `mlir::Liveness` will work for the `acc` region and thus can be
used to perform this analysis.

### Redundant Clause Elimination

The data operations are modeled in a way where data entry operations
look like loads and data exit operations look like stores. Thus these
operations are intended to be optimized in the following ways:
* Be able to eliminate redundant operations such as when an `acc.copyin`
dominates another.
* Be able to hoist/sink such operations out of loops.

## Operations TOC

[include "Dialects/OpenACCDialectOps.md"]

