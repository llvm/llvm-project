# AMDGPU LLVM Extensions for Heterogeneous Debugging <!-- omit in toc -->

- [1. Introduction](#1-introduction)
- [2. High-Level Goals](#2-high-level-goals)
- [3. Motivation](#3-motivation)
- [4. Specification](#4-specification)
  - [4.1. External Definitions](#41-external-definitions)
    - [4.1.1. Well-formed](#411-well-formed)
    - [4.1.2. Type](#412-type)
    - [4.1.3. Value](#413-value)
    - [4.1.4. Location Description](#414-location-description)
  - [4.2. LLVM Debug Information Expressions](#42-llvm-debug-information-expressions)
  - [4.3. LLVM Expression Evaluation Context](#43-llvm-expression-evaluation-context)
  - [4.4. Location Descriptions of LLVM Entities](#44-location-descriptions-of-llvm-entities)
  - [4.5. Metadata](#45-metadata)
    - [4.5.1. DILifetime](#451-dilifetime)
    - [4.5.2. DIObject](#452-diobject)
      - [4.5.2.1. DIVariable](#4521-divariable)
        - [4.5.2.1.1. DILocalVariable](#45211-dilocalvariable)
        - [4.5.2.1.2. DIGlobalVariable](#45212-diglobalvariable)
      - [4.5.2.2. DIFragment](#4522-difragment)
    - [4.5.3. DIExpr](#453-diexpr)
  - [4.6. Intrinsics](#46-intrinsics)
    - [4.6.1. `llvm.dbg.def(metadata, metadata)`](#461-llvmdbgdefmetadata-metadata)
    - [4.6.2. `void @llvm.dbg.kill(metadata)`](#462-void-llvmdbgkillmetadata)
  - [4.7. Global Variable Metadata Attachments](#47-global-variable-metadata-attachments)
    - [4.7.1. !dbg.default !DILifetime](#471-dbgdefault-dilifetime)
  - [4.8. Operations](#48-operations)
    - [4.8.1. `DIOpReferrer(T:type) { -> L:T }`](#481-diopreferrerttype----lt-)
    - [4.8.2. `DIOpArg(N:index, T:type) { -> L:T }`](#482-diopargnindex-ttype----lt-)
    - [4.8.3. `DIOpConstant(T:type V:literal) { -> L:T }`](#483-diopconstantttype-vliteral----lt-)
    - [4.8.4. `DIOpConvert(T':type) { L:T -> L':T' }`](#484-diopconvertttype--lt---lt-)
    - [4.8.5. `DIOpReinterpret(T':type) { L:T -> L:T' }`](#485-diopreinterpretttype--lt---lt-)
    - [4.8.6. `DIOpOffset() { L:T O:U -> L':T }`](#486-diopoffset--lt-ou---lt-)
    - [4.8.7. `DIOpComposite(N:index, T:type) { L[0]:T[0] L[1]:T[1] ... L[N]:T[N] -> L:T }`](#487-diopcompositenindex-ttype--l0t0-l1t1--lntn---lt-)
    - [4.8.8. `DIOpAddrOf(N:addrspace) { L:T -> L':T' }`](#488-diopaddrofnaddrspace--lt---lt-)
    - [4.8.9. `DIOpDeref() { L:T -> L':T' }`](#489-diopderef--lt---lt-)
    - [4.8.10. `DIOpRead() { L:T -> L':T }`](#4810-diopread--lt---lt-)
    - [4.8.11. `DIOpAdd() { L1:T L2:T -> L:T }`](#4811-diopadd--l1t-l2t---lt-)
    - [4.8.12. `DIOpSub() { L1:T L2:T -> L:T }`](#4812-diopsub--l1t-l2t---lt-)
    - [4.8.13. `DIOpMul() { L1:T L2:T -> L:T }`](#4813-diopmul--l1t-l2t---lt-)
    - [4.8.14. `DIOpDiv() { L1:T L2:T -> L:T }`](#4814-diopdiv--l1t-l2t---lt-)
    - [4.8.15. `DIOpShr() { L1:T L2:T -> L:T }`](#4815-diopshr--l1t-l2t---lt-)
    - [4.8.16. `DIOpShl() { L1:T L2:T -> L:T }`](#4816-diopshl--l1t-l2t---lt-)
  - [4.9. Translating to DWARF](#49-translating-to-dwarf)
  - [4.10. Translating to PDB (CodeView)](#410-translating-to-pdb-codeview)
- [5. Examples](#5-examples)
  - [5.1. A variable "x" located in an alloca](#51-a-variable-x-located-in-an-alloca)
  - [5.2. The variable "x" promoted to an SSA register](#52-the-variable-x-promoted-to-an-ssa-register)
  - [5.3. Implicit pointer](#53-implicit-pointer)
  - [5.4. An variable "x" is broken into two scalars](#54-an-variable-x-is-broken-into-two-scalars)
  - [5.5. Example of further decomposition of an already SRoA'd variable](#55-example-of-further-decomposition-of-an-already-sroad-variable)
  - [5.6. Example of multiple live ranges for a single variable](#56-example-of-multiple-live-ranges-for-a-single-variable)
  - [5.7. Example of induction variable](#57-example-of-induction-variable)
  - [5.8. Proven constant](#58-proven-constant)
- [6. Other Ideas](#6-other-ideas)
  - [6.1. Integer fragment IDs](#61-integer-fragment-ids)
    - [6.1.1. A variable "x" is broken into two scalars](#611-a-variable-x-is-broken-into-two-scalars)
    - [6.1.2. Example of further decomposition of an already SRoA'd variable](#612-example-of-further-decomposition-of-an-already-sroad-variable)
    - [6.1.3. Example of multiple live ranges for a fragment](#613-example-of-multiple-live-ranges-for-a-fragment)
- [7. References](#7-references)

# 1. Introduction

As described in the [DWARF Extensions For Heterogeneous Debugging][0] (the
"DWARF extensions"), AMD has been working to support debugging of heterogeneous
programs. This document describes changes to the LLVM representation of debug
information (the "LLVM extensions") required to support the DWARF extensions.
These LLVM extensions continue to support previous versions of the DWARF
standard, including DWARF 5 without extensions, as well as other debug formats
which LLVM currently supports, such as CodeView.

The LLVM extensions do not constitute a direct implementation of all concepts
from the DWARF extensions, although wherever reasonable the most fundamental
aspects are kept identical. The concepts defined in the DWARF extensions which
are used directly in the LLVM extensions with their semantics unchanged are
enumerated in the [External Definitions](#external-definitions) section below.

The most significant departure from the DWARF extensions is in the
consolidation of expression evaluation stack entries. In the DWARF extensions,
each entry on the expression evaluation stack contains either a typed value or
an untyped location description. In the LLVM extensions, each entry on the
expression evaluation stack instead contains a pair of a location description
and a type.

Additionally, the concept of a "generic type", used as a default when a type is
needed but not stated explicitly, is eliminated. Together these changes imply
that the concrete set of operations available differ between the DWARF and LLVM
extensions.

These changes are made to remove redundant representations of semantically
equivalent expressions, which simplifies the work the compiler must do when
updating debug information expressions to reflect code transformations. This is
possible in the LLVM extensions as there is no requirement for backwards
compatibility, nor any requirement that the intermediate representation of debug
information conform to any particular external specification. Consequently we
are able to increase the accuracy of existing debug information, while also
extending the debug information to cover cases which were previously not
described at all.

# 2. High-Level Goals

There are several specific cases where our approach will allow for more
accurate or more complete debug information than would be feasible
with only incremental changes to the existing approach.

* Support describing the location of induction variables. LLVM currently has a
  new implementation of partial support for expressions which depend on
  multiple LLVM values, although it is currently limited exclusively to a
  subset of cases for induction variables. This support is also inherently
  limited as it can only refer directly to LLVM values, not to source variables
  symbolically. This means it is not possible to describe an induction variable
  which, for example, depends on a variable whose location is not static over
  the whole lifetime of the induction variable.
* Support describing the location of arbitrary expressions over scalar-replaced
  aggregate values, even in the face of other dependant expressions. LLVM
  currently must drop debug information when any expression would depend on a
  composite value.
* Support describing all locations of values which are live in multiple machine
  locations at the same instruction. LLVM currently must pick only one such
  location to describe. This means values which are resident in multiple places
  must be conservatively marked read-only, even when they could be read-write
  if all of their locations were reported accurately.
* Accurately support describing the range over which a given location is valid.
  LLVM currently pessimizes debug information as there is no rigorous means to
  limit the range of a described location.
* Support describing the factoring of expressions. This allows features such as
  DWARF procedures to be used to reduce the size of debug information. Factoring
  can also be more convenient for the compiler to describe lexically nested
  information such as program location for inactive lanes in divergent control
  flow.

# 3. Motivation

The original motivation for this proposal was to make the minimum required
changes to the existing LLVM representation of debug information needed to
support the [DWARF Extensions For Heterogeneous Debugging][0]. This involved an
evaluation of the existing debug information for machine locations in LLVM,
which uncovered some hard-to-fix bugs rooted in the incidental complexity and
inconsistency of LLVM's debug intrinsics and expressions.

Attempting to address these bugs in the existing framework proved more difficult
than expected. It became apparent that the shortcomings of the existing solution
were a direct consequence of the complexity, ambiguity, and lack of
composability encountered in DWARF.

With this in mind, we revisited the DWARF extensions to see if they could inform
a more tractable design for LLVM. We had already worked to address the
complexity and ambiguity of DWARF by defining a formalization for its expression
language, and improved the composability by unifying values and location
descriptions on the evaluation stack. Together, these changes also increased the
expressiveness of DWARF. Using similar ideas in LLVM, allowed us to support
additional real world cases, and describe existing cases with greater accuracy.

This led us to start from the DWARF extensions and design a new set of debug
information representations. This was very heavily influenced by prior art in
LLVM, existing RFCs, mailing list discussions, review comments, and bug reports,
without which we would not have been able to make this proposal. Some of the
influences include:

* The use of intrinsics to capture local LLVM values keeps the proposal close to
  the existing implementation, and limits the incidental work needed to support
  it for the reasons outlined in [[LLVMdev] [RFC] Separating Metadata from the
  Value hierarchy][4].
* Support for debug locations which depend on multiple LLVM values is required
  by several optimizations, including expressing induction variables, which is
  the motivation for [D81852 [DebugInfo] Update MachineInstr interface to better
  support variadic DBG_VALUE instructions][5].
* Our solution also generalizes the notion of "fragments" to support composing
  with arbitrary expressions. For example, fragmentation can be represented even
  in the presence of arithmetic operators, as occurs in [D70601 Disallow
  DIExpressions with shift operators from being fragmented][8].
* The desire to support multiple concurrent locations for the same variable is
  described in detail in [[llvm-dev] Proposal for multi location debug info
  support in LLVM IR][6] (continued at [[llvm-dev] Proposal for multi location
  debug info support in LLVM IR][7]) and [Multi Location Debug Info support for
  LLVM][11]. Support for overlapping location list entries was added in DWARF 5.
* Bugs, like those partially worked around in [D57962 [DebugInfo] PR40628:
  Don't salvage load operations][9], often result from passes being unable to
  accurately represent the relationship between source variables. Our approach
  supports encoding that information in debug information in a mechanical way,
  with straightforward semantics.
* Use of `distinct` for our new metadata nodes is motivated by use cases
  similar to those in [[LLVMdev] [RFC] Separating Metadata from the Value
  hierarchy (David Blaikie)][3] where the content of a node is not sufficient
  context to unique it.

Recognizing that the least error prone place to make changes to debug
information is at the point where the underlying code is being transformed, we
biased the representation for this case.

The expression evaluation stack contains uniform pairs of location description
and type, such that all operations have well-defined semantics and no
side-effects on the evaluation of the surrounding expression. These same
semantics apply equally throughout the compiler. This allows for referentially
transparent updates which can be reasoned about in the context of a single
operation and its inputs and outputs, rather than the space of all possible
surrounding operations and dependant expressions.

By eliminating any implicit expression inputs or operations, and constraining
the state space of expressions using well-formedness rules, it is always
unambiguous whether a given transformation is valid and semantics-preserving,
without ever having to consider anything outside of the expression itself.

Designing around a separation of concerns regarding expression modification and
simplification allows each update to the debug information to introduce
redundant or sub-optimal expressions. To address this, an independent
"optimizer" can simplify and canonicalize expressions. As the expression
semantics are well-defined, an "optimizer" can be run without specific
knowledge of the changes made by any one pass or combination of passes.

Incorporating a means to express "factoring", or the definition of one
expression in terms of one or more other expressions, makes "shallow" updates
possible, bounding the work needed for any given update. This factoring is
usually trivial at the time the expression is created, but expensive to infer
later. Factored expressions can result in more compact debug information by
leveraging dynamic calling of DWARF procedures in DWARF 5, and we expect to be
able to use factoring for other purposes, such as debug information for
[divergent control flow][14]. It is possible to statically "flatten" this
factored representation later, if required by the debug information format
being emitted, or if the emitter determines it would be more profitable to do
so.

Leveraging the DWARF extensions as a foundation, the concept of a location
description is used as the fundamental means of recording debug information. To
support this, every LLVM entity which can be referenced by an expression has a
well-defined location description, and is referred to by expressions in an
explicit, referentially transparent manner. This makes updates to reflect
changes in the underlying LLVM representation mechanical, robust, and simple.
Due to factoring, these updates are also more localized, as updates to an
expression are transparently reflected in all dependant expressions without
having to traverse them, or even be aware of their existence.

Without this factoring, any changes to an LLVM entity which is effectively used
as an input to one or more expressions must be "macro-expanded" at the time
they are made, in each place they are referenced. This in turn inhibits the
valid transformations the context-insensitive "optimizer" can safely perform,
as perturbing the macro-expanded expression for an LLVM entity makes it
impossible to reflect future changes to that entity in the expression. Even if
this is considered acceptable, once expressions begin to effectively depend on
other expressions (for example, in the description of induction variables,
where one program object depends on multiple other program objects) there is no
longer a bound on the recursive depth of expressions which must be visited for
any given update, making even simple updates expensive in terms of compiler
resources. Furthermore, this approach requires either a combinatorial explosion
of expressions to describe cases when the live ranges of multiple program
objects are not equal, or the dropping of debug information for all but one
such object. None of these tradeoffs were considered acceptable.

# 4. Specification
## 4.1. External Definitions

Some required concepts are defined outside of this document. We reproduce some
parts of those definitions, along with some expansion on their relationship to
this proposal.

### 4.1.1. Well-formed

The definition of "well-formed" is the one from the section titled
[Well-Formedness in the LLVM Language Reference Manual][10].

### 4.1.2. Type

The definition of "type" is the one from the [LLVM Language Reference
Manual][15].

### 4.1.3. Value

The definition of "value" is the one from the [LLVM Language Reference
Manual][15].

### 4.1.4. Location Description

The definitions of "location description", "single location description", and
"location storage" are the ones from the section titled [DWARF Location
Description][12] in the DWARF Extensions For Heterogeneous Debugging.

A location description can consist of one or more single location descriptions.
A single location description specifies a location storage and bit offset. A
location storage is a linear stream of bits with a fixed size.

The storage encompasses memory, registers, and literal/implicit values.

Zero or more single location descriptions may be valid for a location
description at the same instruction.

## 4.2. LLVM Debug Information Expressions

_[Note: LLVM expressions derive much of their semantics from the DWARF
expressions described in the [AMDGPU Dwarf Extensions][1].]_

LLVM debug information expressions ("LLVM expressions") specify a typed
location. _[Note: Unlike DWARF expressions, they cannot directly describe how to
compute a value. Instead, they are able to describe how to define an implicit
location description for a computed value.]_

If the evaluation of an LLVM expression does not encounter an error, then it
results in exactly one pair of location description and type.

If the evaluation of an LLVM expression encounters an error, the result is an
evaluation error.

If an LLVM expression is not well-formed, then the result is undefined.

The following sections detail the rules for when a LLVM expression is not
well-formed or results in an evaluation error.

## 4.3. LLVM Expression Evaluation Context

An LLVM expression is evaluated in a context that includes the same context
elements as described in [DWARF Expression Evaluation Context][13] with the
following exceptions.  The _current result kind_ is not applicable as all LLVM
expressions are location descriptions.  The _current object_ and _initial stack_
are not applicable as LLVM expressions have no implicit inputs.

## 4.4. Location Descriptions of LLVM Entities

_[TODO: Categorize `MO_ConstantPoolIndex`, `MO_ExternalSymbol`; explicitly
describe all `MachineOperandType`s, including those for which there is no
location description.]_

The notion of location storage is extended to include the abstract LLVM IR
entities of _SSA values_, _stack slots_, _virtual registers_, and _physical
registers_. In each case the location storage conceptually holds the value of
the corresponding entity.

In addition, an implicit address location storage kind is defined. The size of
the storage matches the size of the type for the address.  The value in the
storage is only meaningful when used in its entirety by a `deref` operation,
which yields a location description for the entity that the address references.
_[Note: This is a generalization to the implicit pointer location description
of DWARF 5.]_

Location descriptions can be associated with instances of any of these location
storage kinds.

### LLVM IR SSA Values

The location description of an LLVM IR SSA value `V` specifies a location
storage `LS` and offset `O` which identify the least significant bit of the
object described by `V`. The size of `LS`, minus `O`, is no less than the size
of `V`.

_[Note: The kind of `LS` is unspecified and referentially transparent, but
values of the following kinds generally map to the corresponding kind of
location storage:]_

- `memory location storage`: N/A
- `implicit location storage`: Constant Values, including Global Variables and
  Function Addresses
  - `undefined location storage`: Undefined Values
  - `composite location storage`: N/A
  - `register location storage`: Most Other Values

### LLVM MIR Physical and Virtual Register Operands

The location description of an LLVM MIR Physical or Virtual Register operand
`R` specifies a register location storage `RLS` and offset `O` which identify
the least significant bit of `R`. The size of the `RLS`, minus `O`, is no less
than the size of `R`.

_[Note: The corresponding `MachineOperandType` is `MO_Register`.]_

### LLVM MIR Immediate Operands

The location description of an LLVM MIR Immediate operand `I` specifies an
implicit location storage `ILS` and offset `O` which identify the least
significant bit of `I`. The size of the `ILS`, minus `O`, is no less than the
size of `I`.

_[Note: The corresponding `MachineOperandType`s are: `MO_Immediate`,
`MO_CImmediate`, `MO_FPImmediate`, `MO_GlobalAddress`, `MO_BlockAddress`.]_

### LLVM MIR Frame Index Operands

The location description of an LLVM MIR Frame Index operand `F` specifies a
memory location storage `MLS` with target-specific stack address space and
offset `O` which identify the least significant bit of the stack slot
identified by `F`. The size of the `MLS`, minus `O`, is no less than the size
of the stack slot identified by `F`.

_[Note: The corresponding `MachineOperandType`s are: `MO_FrameIndex`.]_

## 4.5. Metadata

An abstract metadata node exists only to abstractly specify common aspects of
derived node types, and to refer to those derived node types generally.
Abstract node types cannot be created directly.

### 4.5.1. DILifetime

```llvm
distinct !DILifetime(object: !DIObject, location: !DIExpr, argObjects: {!DIObject,...})
```

Represents a lifetime segment of a `DIObject` specified in the required `object`
field.

The required `location` field specifies the expression which evaluates to the
location description of the lifetime segment.

The optional `argObjects` field specifies a tuple of zero or more input
`DIObject`s to the expression. Omitting the `argObjects` field is equivalent to
specifying it to be the empty tuple.

A given `DILifetime` is not well-formed if the `argObjects` tuple contains the
`object`, or if an element is repeated in the `argObjects` tuple.

A given `DILifetime` represent exactly one of the three kinds of lifetime
segments:

* If the `DILifetime` appears as the first argument to exactly one call to the
  `llvm.dbg.def` intrinsic, it specifies a bounded lifetime segment. The call
  to `llvm.dbg.def` is the start of the range covered by the lifetime segment.
  The range extends along all forward control flow paths until either a call to
  a `llvm.dbg.kill` intrinsic which specifies the same `DILifetime`, or to the
  end of an exit basic block.
* If the `DILifetime` appears as a metadata attachment named `dbg.default` on
  exactly one global variable, it specifies a default lifetime segment.
* If the `DILifetime` is never referred to, it specifies an unused lifetime
  segment.

A given `DILifetime` which does not match exactly one above case is not
well-formed.

### 4.5.2. DIObject

A `DIObject` is an abstract metadata node.

Represents the identity of a program object.

Information about the location of a program object is provided by an associated
location description.

The location description for a program object is a function of the current
instruction and a set of associated lifetime segments. A lifetime segment
describes a location description and information about when that location
description is valid.

There are three distinct kinds of lifetime segments:

* Bounded Lifetime Segment_: A lifetime segment with an associated range of
  instructions over which it is valid.
* Default Lifetime Segment_: A lifetime segment which is valid when no bounded
  lifetime segments are valid. There can be at most one default lifetime
  segment for any given program object.
* Unused Lifetime Segment_: A lifetime segment which is always invalid, and so
  does not contribute to the definition of the location description of the
  program object.

For a given instruction, the location description of a program object is:

* If any lifetime segment is not well-formed, the result is undefined.
* If any bounded lifetime segment is valid, then the location description is
  comprised of all of the location descriptions of all valid bounded lifetime
  segments.
* Otherwise, if the program object has a default lifetime segment, then the
  location description is comprised of the location description of that default
  lifetime segment.
* Otherwise, the lifetime description is one undefined location description.

_[Note: When multiple lifetime segments for the same DIObject are active at a
given instruction, it describes the situation where an object exists
simultaneously in more than one place. For example, if a variable exists both
in memory and in a register after the value is spilled but before the register
is clobbered.]_

#### 4.5.2.1. DIVariable

A `DIObject` is an abstract metadata node.

A `DIVariable` is a `DIObject` which represents the identity of a source
variable program object.

##### 4.5.2.1.1. DILocalVariable

A `DILocalVariable` is a `DIVariable` which represents the identity of a local
source variable program object. See [DILocalVariable][16].

##### 4.5.2.1.2. DIGlobalVariable

A `DIGlobalVariable` is a `DIVariable` which represents the identity of a
global source variable program object. See [DIGlobalVariable][17].

#### 4.5.2.2. DIFragment

```llvm
distinct !DIFragment()
```

A `DIFragment` is a `DIObject` which represents the identity of a non-source
variable program object, or a piece of a source or non-source variable program
object.

A non-source variable may be introduced by the compiler. These may be used in
expressions needed for describing debugging information required by the
debugger.

_[Note: In DWARF this can be represented using a DW_TAG_dwarf_procedure DIE.]_

_[Example: An implicit variable needed for calculating the size of a
dynamically sized array.]_

_[Example: An induction variable whose value depends on two source
variables.]_

_[Example: SRoA splits up variables such that there is no single LLVM entity
remaining to describe the location of a source variable.]_

_[Example: Divergent control flow can be described by factoring information
about how to determine active lanes by lexical scope, which results in more
compact debug information.]_

### 4.5.3. DIExpr

```llvm
!DIExpr(DIOp, ...)
```

Represents an expression, which is a sequence of zero or more operations
defined in the following sections.

The evaluation of an expression is in the context of an associated lifetime
segment.

The evaluation of a well-formed expression always yields one typed location
description, which is the only stack entry. If the stack is empty or contains
more than one entry at the end of evaluation, the result is an error.

## 4.6. Intrinsics

_[Note: These intrinsics ultimately define the PC range over which the location
description yielded by a `DILifetime` is active. By walking all such defs/kills
in a module, and collecting their PC ranges, the [DWARF location list][2] for a
given source level variable can be constructed.]_

### 4.6.1. `llvm.dbg.def(metadata, metadata)`

The first argument to `llvm.dbg.def` must be a `DILifetime`, and is the
lifetime begin defined.

The second argument to `llvm.dbg.def` must be a value-as-metadata, and defines
the LLVM entity acting as the referrer of the lifetime segment specified by
the first argument.

A value of `undef` corresponds to the undefined location description, and is
used as the referrer when the expression is not directly IR dependant.

### 4.6.2. `void @llvm.dbg.kill(metadata)`

The first argument to `llvm.dbg.kill` must be a `DILifetime`, and is the
liftime being killed.

Every call to the `llvm.dbg.kill` intrinsic must be reachable from a call to
the `llvm.dbg.def` intrinsic which specifies the same `DILifetime`, otherwise
it is not well-formed.

## 4.7. Global Variable Metadata Attachments

_[Note: Global variables in LLVM exist for the duration of the program, but may
temporarily exist only in a location unique to a given subprogram. To represent
this situation, this metadata attachment allows setting the "default" lifetime
segment for a global variable, which is valid whenever a more specific bounded
lifetime description of the same variable is not available.]_

### 4.7.1. !dbg.default !DILifetime

Specifies that the lifetime segment specified by the `DILifetime` is a default
lifetime segment.

Defines the referrer of that default lifetime segment to be the global variable
to which `dbg.default` is attached.

## 4.8. Operations

> TODO: Maybe operators should specify their input type(s)? It doesn't really
> match what DWARF does currently, and we can't trivially use them to enforce
> anything via e.g. debug asserts. Because the expression language is an
> arbitrary stack, in general we have to evaluate the whole expression to
> understand the inputs to a given operation.

In the below definitions, the operator `sizeof` computes the size in bits of
the given LLVM type, rather than the size in bytes.

Each definition begins with a specification which describes the parameters to
the operation, the entries it pops from the stack, and the entries it pushes on
the stack. The specification is accepted by the following modified BNF grammer,
where `[]` denotes character classes, `*` denotes zero-or-more repetitions of a
term, and `+` denotes one-or-more repetitions of a term.

```bnf
            <specification> ::= <syntax> <stack-effects>

      <operation-identifer> ::= [a-z_]+
        <binding-identifer> ::= [A-Z][A-Z']*

                   <syntax> ::= <operation-identifier> "(" <parameter-binding-list> ")"
   <parameter-binding-list> ::= "" | <parameter-binding>
                                   | <parameter-binding> "," <paramter-binding-list>
                                   | <parameter-binding> " " <paramter-binding-list>
        <parameter-binding> ::= <binding-identifer> ":" <paramter-binding-kind>
   <parameter-binding-kind> ::= "type" | "index" | "literal" | "addrspace"

            <stack-effects> ::= "{" <stack-binding-list> "->" <stack-binding-list> "}"
       <stack-binding-list> ::= "" | <stack-binding> | <stack-binding> " " <stack-binding-list>
            <stack-binding> ::= <binding-identifier> ":" <binding-identifier>
```

Each `<binding-identifier>` identifies a metasyntactic variable, and either
binds that variable to an entity if this is the first mention of the
identifier, or refers to the entity bound to the identifier, otherwise.

The `<syntax>` describes the concrete syntax of the operation in an LLVM
expression as part of LLVM IR. If an LLVM expression operation does not conform
to the syntax, the result is an error.

The `<parameter-binding-list>` defines positional parameters to the operation.
Each parameter in the list has a `<binding-identifer>` which binds to the
argument passed via the parameter, and a `<parameter-binding-kind>` which
defines the kind of arguments accepted by the parameter.

The possible parameter kinds are:

* `type`: An LLVM first class type.
* `index`: A non-negative literal integer.
* `literal`: An LLVM literal value expression.
* `addrspace`: An LLVM target-specific address space identifier.

> TODO: Define the concrete syntaxes of each parameter kind.

The `<stack-effects>` describe the effect of the operation on the stack. The
first `<stack-binding-list>` describes the "inputs" to the operation, which are
the entries it pops from the stack. The second `<stack-binding-list>` describes
the "outputs" of the operation, which are the entries it pushes onto the stack.

If the execution of an operation would require more entries be popped from the
stack than are present on the stack, the result is an error.

Each `<stack-binding>` is a pair of `<binding-identifier>`s. The first
`<binding-identifier>` binds to the location description of the stack entry.
The second `<binding-identifier>` binds to the type of the stack entry.

A reference to a previously bound (when read left-to-right) metasyntactic
variable is an assertion that the referenced entities are identical. For
parameters and inputs this states a requirement/precondition of the operation
required to be well-formed, for outputs this guarantees an invariant of the
output.

The remaining body of the definition for an operation may reference all of the
bound metasyntactic variable identifiers from the specification, and may
define additional ones following the same left-to-right semantics.

### 4.8.1. `DIOpReferrer(T:type) { -> L:T }`

`L` is the location description of the referrer `R` of the associated lifetime
segment.

`sizeof(T)` must equal `sizeof(R)`, otherwise the expression is not well-formed.

### 4.8.2. `DIOpArg(N:index, T:type) { -> L:T }`

`L` is the location description of the `N`th input `DIObject`, `I`, of the
associated lifetime segment.

`sizeof(T)` must equal `sizeof(I)`, otherwise the expression is not well-formed.

_[Note: As with any location description, the location description pushed by
`arg` may consist of multiple single location descriptions. For example, this
will occur if the `DIObject` referred to has more than one bounded lifetime
segment active. By definition these must all describe the same object. This
implies that when reading from them, any of the single location descriptions
may be chosen, whereas when writing to them the write is performed into each
single location description.]_

### 4.8.3. `DIOpConstant(T:type V:literal) { -> L:T }`

`V` is a literal value of type `T`.

`L` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V` and size
`sizeof(T)`.

### 4.8.4. `DIOpConvert(T':type) { L:T -> L':T' }`

Creates a value `V` of type `T'` by reading `sizeof(T)` bits from `L` and
updating them according to the conversion from type `T` to type `T'`.

> TODO: Define the possible conversions and their semantics in terms of `T` and
`T'`.

`L'` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V` and size
`sizeof(T')`.

### 4.8.5. `DIOpReinterpret(T':type) { L:T -> L:T' }`

`sizeof(T')` must be less than or equal to `sizeof(T)`, otherwise the
expression is not well-formed.

### 4.8.6. `DIOpOffset() { L:T O:U -> L':T }`

`L'` is `L`, but updated by adding `VO` to its offset. `VO` is the value
obtained by reading `sizeof(U)` bits from `O`.

### 4.8.7. `DIOpComposite(N:index, T:type) { L[0]:T[0] L[1]:T[1] ... L[N]:T[N] -> L:T }`

> TODO: Decribe the "variadic" bindings used here, even if informally.

`L` comprises one complete composite location description with `N` parts. The
`M`th part of `L` specifies location description `L[M]`.

### 4.8.8. `DIOpAddrOf(N:addrspace) { L:T -> L':T' }`

`L'` comprises one implicit address location description `IAL`. `IAL` specifies
implicit address location storage `IALS` and offset `0`. `IALS` has addressed
location description `L` and address space `N`.

`T'` is: pointer to `T` in address space `N`.

### 4.8.9. `DIOpDeref() { L:T -> L':T' }`

`T'` is the pointee type of `T`.

`L'` comprises one memory location description `MLD`. `MLD` specifies offset
`A` and the memory location storage corresponding to address space `N`. `N` is
the address space of the pointer type `T`. `A` is the value obtained by reading
`sizeof(T)` bits from `L`.

However, if the bits that would be read from `L` comprise the complete, ordered
contents of one implicit address location description `IAL`, then `L'` is
instead the addressed location description of `IAL`.

Otherwise, the containing expression is not well-formed.

### 4.8.10. `DIOpRead() { L:T -> L':T }`

`L'` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V` and size
`sizeof(T)`.

`V` is the value of type `T` obtained by reading `sizeof(T)` bits from `L`.

### 4.8.11. `DIOpAdd() { L1:T L2:T -> L:T }`

`V1` is the value of type `T` obtained by reading `sizeof(T)` bits from `L1`.
`V2` is the value of type `T` obtained by reading `sizeof(T)` bits from `L2`.

`L` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V1 + V2` and size
`sizeof(T)`.

_[Note: Define overflow and any other operation-specific cases of interest. May
need variants for different behavior to match DWARF?]_

### 4.8.12. `DIOpSub() { L1:T L2:T -> L:T }`

`V1` is the value of type `T` obtained by reading `sizeof(T)` bits from `L1`.
`V2` is the value of type `T` obtained by reading `sizeof(T)` bits from `L2`.

`L` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V1 - V2` and size
`sizeof(T)`.

### 4.8.13. `DIOpMul() { L1:T L2:T -> L:T }`

`V1` is the value of type `T` obtained by reading `sizeof(T)` bits from `L1`.
`V2` is the value of type `T` obtained by reading `sizeof(T)` bits from `L2`.

`L` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V1 * V2` and size
`sizeof(T)`.

### 4.8.14. `DIOpDiv() { L1:T L2:T -> L:T }`

`V1` is the value of type `T` obtained by reading `sizeof(T)` bits from `L1`.
`V2` is the value of type `T` obtained by reading `sizeof(T)` bits from `L2`.

`L` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V1 / V2` and size
`sizeof(T)`.

### 4.8.15. `DIOpShr() { L1:T L2:T -> L:T }`

`V1` is the value of type `T` obtained by reading `sizeof(T)` bits from `L1`.
`V2` is the value of type `T` obtained by reading `sizeof(T)` bits from `L2`.

`L` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V1 >> V2` and size
`sizeof(T)`.

### 4.8.16. `DIOpShl() { L1:T L2:T -> L:T }`

`V1` is the value of type `T` obtained by reading `sizeof(T)` bits from `L1`.
`V2` is the value of type `T` obtained by reading `sizeof(T)` bits from `L2`.

`L` comprises one implicit location description `IL`. `IL` specifies implicit
location storage `ILS` and offset `0`. `ILS` has value `V1 << V2` and size
`sizeof(T)`.

## 4.9. Translating to DWARF

> TODO: work through algorithm for actually computing DWARF location
> descriptions and loclists
>
> * Define rule for implicit pointers (addrof applied to a referrer)
>   * Look for a compatible, existing program object
>   * If not, generate an artificial one
>   * This could be bubbled up to DWARF itself, to allow implicits to hold
>     arbitrary location descriptions, eliminating the need for the artifical
>     variable, and make translation simpler.

## 4.10. Translating to PDB (CodeView)

> TODO

# 5. Examples

Examples which need meta-syntactic variables will prefix them with an
appropriate sigil to try to concisely give some context for them. The prefix
sigils are:

| Sigil | Meaning
|-------|---------
|   %   | SSA IR Value
|   $   | Non-SSA MIR Register (i.e. post phi-elim)
|   #   | Arbitrary literal constant

The syntax used in examples attempts to match LLVM IR/MIR as closely as
possible, with the only new syntax required being that of the expression
language.

## 5.1. A variable "x" located in an alloca

The frontend will generate `alloca`s for every variable, and can trivially
insert a single `DILifetime` covering the whole body of the function, with the
expression `DIExpr(DIOpReferrer(), DIOpDeref(<stack-aspace>))`, referring to
the alloca. Walking all of the debug intrinsics provides enough information to
generate the loclist.

```llvm
%addr.x = alloca i64, addrspace(5)
call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %addr.x)
store i64* %addr.x, ...
...
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref()))
```

## 5.2. The variable "x" promoted to an SSA register

The promotion semantically removes one level of indirection, and
correspondingly in the debug expressions for which the alloca being replaced
was the referrer, an additional `DIOpAddrOf(N)` is needed.

An example is mem2reg where an alloca can be replaced with an SSA value:

```llvm
%x = i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
...
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5), DIOpDeref()))
```

The canonical form of this is then just `DIOpReferrer(i64)` as the pair of
`DIOpAddrOf(N), DIOpDeref()` cancel out:

```llvm
%x = i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
...
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
```

## 5.3. Implicit pointer

The transformation for removing a level of indirection is always to add an
`DIOpAddrOf(N)`, which may result in a location description for a pointer to a
non-memory object.

> TODO: Well-formedness rule for mismatching address spaces.

```c
int x = ...;
int *p = &x;
return *p;
```

> TODO: Should the dbg.def follow the alloca or the store? We have no way to
> explain that something has a location, but the location doesn't hold a
> meaningful value yet.

```llvm
%x = alloca i64, addrspace(5)
call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x)
store i64 addrspace(5)* %x, i64 ...
%p = alloca i64*, addrspace(5)
call void @llvm.dbg.def(metadata !4, metadata i64 addrspace(5)* addrspace(5)* %p)
store i64 addrspace(5)* addrspace(5)* %p, i64 addrspace(5)* %x
load i64 addrspace(5)* %0, i64 addrspace(5)* addrspace(5)* %p
load i64 %1, i64 addrspace(5)* %0
ret i64 %1
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref()))
!3 = !DILocalVariable("p", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64 addrspace(5)* addrspace(5)*), DIOpDeref()))
```

First round of mem2reg:

```llvm
%x = alloca i64, addrspace(5)
store i64 addrspace(5)* %x, i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x)
%p = i64 addrspace(5)* %x
call void @llvm.dbg.def(metadata !4, metadata i64 addrspace(5)* %p)
load i64 %0, i64 addrspace(5)* %p
return i64 %0
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref()))
!3 = !DILocalVariable("p", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpAddrOf(5), DIOpDeref()))
```

Simplified:

```llvm
%x = alloca i64, addrspace(5)
store i64 addrspace(5)* %x, i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x)
call void @llvm.dbg.def(metadata !4, metadata i64 addrspace(5)* %x)
load i64 %0, i64 addrspace(5)* %x
return i64 %0
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref()))
!3 = !DILocalVariable("p", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*)))
```

Second round of mem2reg:

```llvm
%x = i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
call void @llvm.dbg.def(metadata !4, metadata i64 %x)
%0 = i64 %x
return i64 %0
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5), DIOpDeref()))
!3 = !DILocalVariable("p", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5)))
```

Simplified:

```llvm
%x = i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
return i64 %x
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
!3 = !DILocalVariable("p", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5)))
```

If `%x` is eliminated entirely and replaced with a constant:

```llvm
call void @llvm.dbg.def(metadata !2, metadata i1 undef)
call void @llvm.dbg.def(metadata !4, metadata i1 undef)
return i64 ...
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpConstant(i64 ...)))
!3 = !DILocalVariable("p", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpConstant(i64 ...), DIOpAddrOf(5)))
```

## 5.4. An variable "x" is broken into two scalars

When a transformation decomposes one location into multiple distinct ones, it
must follow all `def` intrinsics to the `DILifetime`s referencing the original
location and update the expression and positional arguments such that:

* All instances of `DIOpReferrer()` in the original expression are replaced
  with the appropriate composition of all the new location pieces, now encoded
  via `DIOpArg()` operations and input `DIObject`s.
* Those location pieces are represented by new `DIFragment`s, one per new
  location, each with appropriate `DILifetime`s referenced by new `def` and
  `kill` intrinsics.

It is assumed that any transformation capable of doing the decomposition in the
first place must have all of this information available, and the structure of
the new intrinsics and metadata avoids any costly operations during
transformations. This update is also "shallow", in that only the `DILifetime`
which is immediately referenced by the relevant `def`s need to be updated, as
the result is referentially transparent to any other dependant `DILifetime`s.

```llvm
%x.lo = i32 ...
call void @llvm.dbg.def(metadata !4, metadata i32 %x.lo)
...
%x.hi = i32 ...
call void @llvm.dbg.def(metadata !6, metadata i32 %x.hi)
...
call void @llvm.dbg.kill(metadata !6)
call void @llvm.dbg.kill(metadata !4)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(0, i32), DIOpArg(1, i32), DIOpComposite(2, i64)), !3, !5)
!3 = distinct !DIFragment()
!4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i32)))
!5 = distinct !DIFragment()
!6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpReferrer(i32)))
```

## 5.5. Example of further decomposition of an already SRoA'd variable

An example to demonstrate the "shallow update" property is to take the above
IR and subdivide `%x.hi` again:

```llvm
%x.lo = i32 ...
call void @llvm.dbg.def(metadata !4, metadata i32 %x.lo)
%x.hi.lo = i16 ...
call void @llvm.dbg.def(metadata !8, metadata i16 %x.hi.lo)
%x.hi.hi = i16 ...
call void @llvm.dbg.def(metadata !10, metadata i16 %x.hi.hi)
...
call void @llvm.dbg.kill(metadata !10)
call void @llvm.dbg.kill(metadata !8)
call void @llvm.dbg.kill(metadata !4)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(0, i32), DIOpArg(1, i32), DIOpComposite(2, i64)), !3, !5)
!3 = distinct !DIFragment()
!4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i32)))
!5 = distinct !DIFragment()
!6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpArg(0, i16), DIOpArg(1, i16), DIOpComposite(2, i32)), !7, !9)
!7 = distinct !DIFragment()
!8 = distinct !DILifetime(object: !7, location: !DIExpr(DIOpReferrer(i16)))
!9 = distinct !DIFragment()
!10 = distinct !DILifetime(object: !9, location: !DIExpr(DIOpReferrer(i16)))
```

Note that the expression for the original source variable "x" did not need to
be changed, as it is defined in terms of the `DIFragment`, the identity of
which is never changed after it is created.

## 5.6. Example of multiple live ranges for a single variable

Once out of SSA, or even while in SSA via memory, there may be multiple re-uses
of the same storage for completely disparate variables, and disjoint and/or
overlapping lifetimes for any single variable. This is modeled naturally by
maintaining `def`s and `kill`s for these live ranges independently at e.g.
definitions and clobbers.

```llvm
$r0 = MOV ...
DBG_DEF !2, $r0
...
SPILL %frame.index.0, $r0
DBG_DEF !3, %frame.index.0
...
$r0 = MOV ; clobber
DBG_KILL !2
DBG_DEF !6, $r0
...
$r1 = MOV ...
DBG_DEF !4, $r1
...
DBG_KILL !6
DBG_KILL !4
DBG_KILL !3
DBG_KILL !2
RETURN

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i32)))
!3 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i32)))
!4 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i32)))
!5 = !DILocalVariable("y", ...)
!6 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(i32)))
```

In this example, $r0 is referred to by disjoint `DILifetime`s for different
variables. There is also a point where multiple `DILifetime`s for the same
variable are live.

The first point implies the need for intrinsics/psuedo-instructions to define
the live range, as simply referring to an LLVM entity doesn't provide enough
information to reconstruct the live range.

The second point is needed to accurately represent cases where, e.g. a variable
lives in both a register and in memory. The current
intrinsics/pseudo-instructions do not have the notion of live ranges for source
variables, and simply throw away at least one of the true lifetimes in these
cases.

## 5.7. Example of induction variable

Starting with some program:

```llvm
%x = i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
...
%y = i64 ...
call void @llvm.dbg.def(metadata !4, i64 %y)
...
%i = i64 ...
call void @llvm.dbg.def(metadata !6, metadata i64 %z)
...
call void @llvm.dbg.kill(metadata !6)
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
!3 = !DILocalVariable("y", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64)))
!5 = !DILocalVariable("i", ...)
!6 = !DILifetime(object: !5, location: !DIExpr(DIOpReferrer(i64)))
```

If analysis proves `i` over some range is always equal to `x + y`, the storage
for `i` can be eliminated, and it can be materialized at every use. The
corresponding change needed in our debug info is:

```llvm
%x = i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
...
%y = i64 ...
call void @llvm.dbg.def(metadata !4, metadata i64 %y)
...
call void @llvm.dbg.def(metadata !6, metadata i1 undef)
...
call void @llvm.dbg.kill(metadata !6)
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
!3 = !DILocalVariable("y", ...)
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64)))
!5 = !DILocalVariable("i", ...)
!6 = !DILifetime(object: !5, location: !DIExpr(DIOpArg(0, i64), DIOpArg(1, i64), DIOpAdd()), !1, !3)
```

For the given range, the value of `i` is computable so long as both `x` and `y`
are live, the determination of which is left until the DI backend (e.g. for old
DWARF or for other DI formats), or until runtime (e.g. for DWARF with
DW_OP_call and subroutines). During compilation this representation allows all
updates to maintain the debug info efficiently by making updates "shallow".

In other cases this can allow the debugger to provide locations for part of a
source variable, even when other parts are not available. This may be the case
if a struct with many fields is broken up during SRoA and the lifetimes of each
piece diverge.

## 5.8. Proven constant

As a very similar example to the above induction variable case (in terms of the
updates needed in the debug info) the case where a variable is proven to be
a statically known constant over some range turns the following:

```llvm
%x = i64 ...
call void @llvm.dbg.def(metadata !2, metadata i64 %x)
...
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
```

into:

```llvm
call void @llvm.dbg.def(metadata !2, metadata i1 undef)
...
call void @llvm.dbg.kill(metadata !2)

!1 = !DILocalVariable("x", ...)
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpConstant(i64 ...)))
```

> TODO:
>
> * divergent control flow case
> * simultaneous lifetimes in multiple places
> * file scope globals

```llvm
@g = i64 !dbg.default !2

!1 = !DIGlobalVariable("g")
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
```

Becomes:

```llvm
@g.lo = i32 !dbg.default !4
@g.hi = i32 !dbg.default !6

!1 = !DIGlobalVariable("g")
!2 = !DILifetime(object: !1, location: !DIExpr(DIOpArg(0, i32), DIOpArg(1, i32), DIOpComposite(2, i64)), !3, !5)
!3 = !DIFragment()
!4 = !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i32)))
!5 = !DIFragment()
!6 = !DILifetime(object: !5, location: !DIExpr(DIOpReferrer(i32)))

!dbg.default is "hidden" when any other lifetime is in effect. Allows e.g. a
function to override the location of a global over some range without needing
to "kill" and "def" a global lifetime.
```

> TODO: LDS variables, one variable but multiple kernels with distinct
> lifetimes, is that possible in LLVM?
>
> We could allow the def intrinsic to refer to a global, and use that to define
> live ranges which live in functions and refer to storage outside of the
> function.

> TODO: work through CSE case, don't want to drop when not necessary

Example from
[https://bugs.llvm.org/show_bug.cgi?id=40628](https://bugs.llvm.org/show_bug.cgi?id=40628)

```c
    int
    foo(int *bar, int arg, int more)
    {
      int redundant = *bar;
      int loaded = *bar;
      arg &= more + loaded;

      *bar = 0;

      return more + *bar;
    }

    int
    main() {
      int lala = 987654;
      return foo(&lala, 1, 2);
    }
```

Which after SROA+mem2reg becomes:

```llvm
; Function Attrs: noinline nounwind uwtable
define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32* %bar, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %arg, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %more, metadata !15, metadata !DIExpression()), !dbg !18
  %0 = load i32, i32* %bar, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %0, metadata !16, metadata !DIExpression()), !dbg !18
  %1 = load i32, i32* %bar, align 4, !dbg !24, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %1, metadata !17, metadata !DIExpression()), !dbg !18
  %add = add nsw i32 %more, %1, !dbg !25
  %and = and i32 %arg, %add, !dbg !26
  call void @llvm.dbg.value(metadata i32 %and, metadata !14, metadata !DIExpression()), !dbg !18
  store i32 0, i32* %bar, align 4, !dbg !27, !tbaa !20
  %2 = load i32, i32* %bar, align 4, !dbg !28, !tbaa !20
  %add1 = add nsw i32 %more, %2, !dbg !29
  ret i32 %add1, !dbg !30
}
```

And previously led to:

```llvm
define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32* %bar, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %arg, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %more, metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32* %bar, metadata !16, metadata !DIExpression(DW_OP_deref)), !dbg !18
  %0 = load i32, i32* %bar, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !DIExpression()), !dbg !18
  %add = add nsw i32 %more, %0, !dbg !24
  call void @llvm.dbg.value(metadata i32 undef, metadata !14, metadata !DIExpression()), !dbg !18
  store i32 0, i32* %bar, align 4, !dbg !25, !tbaa !20
  ret i32 %more, !dbg !26
}
```

But now becomes (conservatively):

```llvm
define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32* %bar, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %arg, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %more, metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 undef, metadata !16, metadata !DIExpression()), !dbg !18
  %0 = load i32, i32* %bar, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !DIExpression()), !dbg !18
  %add = add nsw i32 %more, %0, !dbg !24
  call void @llvm.dbg.value(metadata i32 undef, metadata !14, metadata !DIExpression()), !dbg !18
  store i32 0, i32* %bar, align 4, !dbg !25, !tbaa !20
  ret i32 %more, !dbg !26
}
```

Effectively at the point of the CSE eliminating the load, it conservatively
marks the source variable `redundant` as optimized out.

It seems like the semantics that CSE really wants to encode in the debug
intrinsics is that, after the point at which the common load occurs, the
location for both `redundant` and `loaded` is `%0`, and that they are both
read-only. It seems like it must prove this to combine them, and if it can only
combine them over some range it can insert additional live ranges to describe
their separate locations outside of that range. The implicit pointer example is
further evidence of why this may need to be the case, because at the time we
create the implicit pointer we don't know which source variable to bind to in
order to get the multiple lifetimes in this design.

This seems to be supported by the fact that even in current LLVM trunk, with
the more-conservative change to mark the `redundant` variable as `undef` in the
above case, modifying `redundant` after the load results in both `redundant`
and `loaded` referring to the same location, and both being read-write. A
modification of `redundant` before the use of `load` is permitted, and then
causes unexpected behavior.

```c
int
foo(int *bar, int arg, int more)
{
  int redundant = *bar;
  int loaded = *bar;
  arg &= more + loaded; // a store to redundant here affects loaded

  *bar = redundant;
  redundant = 1;

  return more + *bar;
}

int
main() {
  int lala = 987654;
  return foo(&lala, 1, 2);
}
```

After Early CSE:

```llvm
define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32* %bar, metadata !14, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %arg, metadata !15, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %more, metadata !16, metadata !DIExpression()), !dbg !19
  %0 = load i32, i32* %bar, align 4, !dbg !20, !tbaa !21
  call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %0, metadata !18, metadata !DIExpression()), !dbg !19
  %add = add nsw i32 %more, %0, !dbg !25
  call void @llvm.dbg.value(metadata i32 undef, metadata !15, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 1, metadata !17, metadata !DIExpression()), !dbg !19
  ret i32 %add, !dbg !26
}
```

Note: To see the result, i386 is required; x86_64 seems to do even more optimization
which eliminates both `loaded` and `redundant`.

> TODO: go over every :ASDF: and make sure we handle it

Mostly done, there are still some unique places I had lumped together before,
but I believe we should do as well or better at the ones I've looked at.

> TODO: SSA -> stack slot

```llvm
%x = i32 ...
call void @llvm.dbg.def(metadata !1, metadata i32 %x)
...
call void @llvm.dbg.kill(metadata !1)

!0 = !DILocalVariable("x")
!1 = !DILifetime(object: !0, location: !DIExpr(DIOpReferrer(i32)))
```

spill %x:

```llvm
%x.addr = alloca i32, addrspace(5)
store i32* %x.addr, ...
call void @llvm.dbg.def(metadata !1, metadata i32 *%x)
...
call void @llvm.dbg.kill(metadata !1)

!0 = !DILocalVariable("x")
!1 = !DILifetime(object: !0, location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref()))
```

> TODO: stack slot -> register

> TODO: register -> stack slot

> TODO: make sure the non-SSA MIR form works with our def/kill scheme, and
> additionally confirm why we don't seem to need the work upstream that is
> trying to move to referring to an instruction rather than a register?
>
> [https://lists.llvm.org/pipermail/llvm-dev/2020-February/139440.html](https://lists.llvm.org/pipermail/llvm-dev/2020-February/139440.html)

> TODO: understand how this compares to what GCC is doing?

# 6. Other Ideas
## 6.1. Integer fragment IDs

_[Note: This was just a quick jotting-down of one idea for eliminating the need
for a distincit `DIFragment` to represent the identity of fragments.]_

### 6.1.1. A variable "x" is broken into two scalars

```llvm
%x.lo = i32 ...
call void @llvm.dbg.def(metadata i32 %x.lo, metadata !4)
...
%x.hi = i32 ...
call void @llvm.dbg.def(metadata i32 %x.hi, metadata !6)
...
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !6)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(var 0, var 1, composite 2))
!3 = distinct !DILifetime(object: 0, location: !DIExpr(referrer))
!4 = distinct !DILifetime(object: 1, location: !DIExpr(referrer))
```

### 6.1.2. Example of further decomposition of an already SRoA'd variable

```llvm
%x.lo = i32 ...
call void @llvm.dbg.def(metadata i32 %x.lo, metadata !3)
%x.hi.lo = i16 ...
call void @llvm.dbg.def(metadata i16 %x.hi.lo, metadata !5)
%x.hi.hi = i16 ...
call void @llvm.dbg.def(metadata i16 %x.hi.hi, metadata !6)
...
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !8)
call void @llvm.dbg.kill(metadata !10)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(var 0, var 1, composite 2))
!3 = distinct !DILifetime(object: 0, location: !DIExpr(referrer))
!4 = distinct !DILifetime(object: 1, location: !DIExpr(var 2, var 3, composite 2))
!5 = distinct !DILifetime(object: 2, location: !DIExpr(referrer))
!6 = distinct !DILifetime(object: 3, location: !DIExpr(referrer))
```

### 6.1.3. Example of multiple live ranges for a fragment

```llvm
%x.lo.0 = i32 ...
call void @llvm.dbg.def(metadata i32 %x.lo, metadata !3)
...
call void @llvm.dbg.kill(metadata !3)
%x.lo.1 = i32 ...
call void @llvm.dbg.def(metadata i32 %x.lo, metadata !4)
%x.hi.lo = i16 ...
call void @llvm.dbg.def(metadata i16 %x.hi.lo, metadata !6)
%x.hi.hi = i16 ...
call void @llvm.dbg.def(metadata i16 %x.hi.hi, metadata !7)
...
call void @llvm.dbg.kill(metadata !4)
call void @llvm.dbg.kill(metadata !6)
call void @llvm.dbg.kill(metadata !7)

!1 = !DILocalVariable("x", ...)
!2 = distinct !DILifetime(object: !1, location: !DIExpr(var 0, var 1, composite 2))
!3 = distinct !DILifetime(object: 0, location: !DIExpr(referrer))
!4 = distinct !DILifetime(object: 0, location: !DIExpr(referrer))
!5 = distinct !DILifetime(object: 1, location: !DIExpr(var 2, var 3, composite 2))
!6 = distinct !DILifetime(object: 2, location: !DIExpr(referrer))
!7 = distinct !DILifetime(object: 3, location: !DIExpr(referrer))
```

# 7. References

1. [DWARF Extensions For Heterogeneous Debugging][0]
2. [AMDGPU Dwarf Extensions][1]
3. [DWARF location list][2]
4. [[LLVMdev] [RFC] Separating Metadata from the Value hierarchy (David Blaikie)][3]
5. [[LLVMdev] [RFC] Separating Metadata from the Value hierarchy][4]
6. [D81852 [DebugInfo] Update MachineInstr interface to better support variadic
   DBG_VALUE instructions][5]
7. [[llvm-dev] Proposal for multi location debug info support in LLVM IR][6]
8. [[llvm-dev] Proposal for multi location debug info support in LLVM IR][7]
9. [D70601 Disallow DIExpressions with shift operators from being fragmented][8]
10. [D57962 [DebugInfo] PR40628: Don't salvage load operations][9]
11. [Well-Formedness in the LLVM Language Reference Manual][10]
12. [Multi Location Debug Info support for LLVM][11]
13. [DWARF Location Description][12]
14. [DWARF Expression Evaluation Context][13]
15. [divergent control flow][14]
16. [LLVM Language Reference Manual][15]
17. [DILocalVariable][16]
18. [DIGlobalVariable][17]

[0]: https://llvm.org/docs/AMDGPUDwarfExtensionsForHeterogeneousDebugging.html
[1]: https://llvm.org/docs/AMDGPUDwarfExtensionsForHeterogeneousDebugging.html#dwarf-expressions
[2]: https://llvm.org/docs/AMDGPUDwarfExtensionsForHeterogeneousDebugging.html#dwarf-location-list-expressions
[3]: https://lists.llvm.org/pipermail/llvm-dev/2014-November/078656.html
[4]: https://lists.llvm.org/pipermail/llvm-dev/2014-November/078682.html
[5]: https://reviews.llvm.org/D81852
[6]: https://lists.llvm.org/pipermail/llvm-dev/2015-December/093535.html
[7]: https://lists.llvm.org/pipermail/llvm-dev/2016-January/093627.html
[8]: https://reviews.llvm.org/D70601
[9]: https://reviews.llvm.org/D57962
[10]: https://llvm.org/docs/LangRef.html#well-formedness
[11]: https://gist.github.com/Keno/480b8057df1b7c63c321
[12]: https://llvm.org/docs/AMDGPUDwarfExtensionsForHeterogeneousDebugging.html#dwarf-location-description
[13]: https://llvm.org/docs/AMDGPUDwarfExtensionsForHeterogeneousDebugging.html#dwarf-expression-evaluation-context
[14]: https://llvm.org/docs/AMDGPUUsage.html#dw-at-llvm-lane-pc
[15]: https://llvm.org/docs/LangRef.html
[16]: https://llvm.org/docs/LangRef.html#dilocalvariable
[17]: https://llvm.org/docs/LangRef.html#diglobalvariable
