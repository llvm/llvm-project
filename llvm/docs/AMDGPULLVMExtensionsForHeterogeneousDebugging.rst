===================================================
AMDGPU LLVM Extensions for Heterogeneous Debugging
===================================================

.. contents::
   :local:

.. warning::

   This section describes **provisional support** for AMDGPU LLVM debug
   information that is not currently fully implemented and is subject to change.

Introduction
============

As described in the :doc:`AMDGPUDwarfExtensionsForHeterogeneousDebugging` (the
“DWARF extensions”), AMD has been working to support debugging of heterogeneous
programs. This document describes changes to the LLVM representation of debug
information (the “LLVM extensions”) required to support the DWARF extensions.
These LLVM extensions continue to support previous versions of the DWARF
standard, including DWARF 5 without extensions, as well as other debug formats
which LLVM currently supports, such as CodeView.

The LLVM extensions do not constitute a direct implementation of all concepts
from the DWARF extensions, although wherever reasonable the fundamental aspects
were kept identical. The concepts defined in the DWARF extensions which are used
directly in the LLVM extensions with their semantics unchanged are enumerated in
the :ref:`amdgpu-llvm-debug-external-definitions` section below.

A significant departure from the DWARF extensions is in the consolidation of
expression evaluation stack entries. In the DWARF extensions, each entry on the
expression evaluation stack contains either a typed value or an untyped location
description. In the LLVM extensions, each entry on the expression evaluation
stack instead contains a pair of a location description and a type.

Additionally, the concept of a “generic type”, used as a default when a type is
needed but not stated explicitly, is eliminated. Together, these changes imply
that the concrete set of operations available differ between the DWARF and LLVM
extensions.

These changes were made to remove redundant representations of semantically
equivalent expressions, which can simplify the compiler’s work in updating debug
information expressions to reflect code transformations. The LLVM extensions’
changes are possible as LLVM has no requirement for backwards compatibility, nor
any requirement that the intermediate representation of debug information
conform to any particular external specification. Consequently, the LLVM
extensions are able to increase the accuracy of existing debug information,
while also extending the debug information to cover cases which were previously
not described at all.

High-Level Goals
================

There are several specific cases where the LLVM extensions’ approach can allow
for more accurate or more complete debug information than would be feasible with
only incremental changes to the existing approach.

-  Support describing the location of induction variables. LLVM currently has a
   new implementation of partial support for an expression which depends on
   multiple LLVM values, although it is currently limited exclusively to a
   subset of cases for induction variables. This support is also inherently
   limited as it can only refer directly to LLVM values, not to source variables
   symbolically. This means it is not possible to describe an induction variable
   which, for example, depends on a variable whose location is not static over
   the whole lifetime of the induction variable.
-  Support describing the location of arbitrary expressions over scalar-replaced
   aggregate values, even in the face of other dependent expressions. LLVM
   currently drops debug information when any expression would depend on a
   composite value.
-  Support describing all locations of values which are live in multiple machine
   locations at the same instruction. LLVM currently picks only one such
   location to describe. This means values which are resident in multiple places
   need to be conservatively marked read-only, even when they could be
   read-write if all of their locations were reported accurately.
-  Accurately support describing the range over which a given location is
   active. LLVM currently pessimizes debug information as there is no rigorous
   means to limit the range of a described location.
-  Support describing the factoring of expressions. This allows features such as
   DWARF procedures to be used to reduce the size of debug information.
   Factoring can also be more convenient for the compiler to describe lexically
   nested information such as program location for inactive lanes in divergent
   control flow.

Motivation
==========

The original motivation for the LLVM extensions was to make the minimum required
changes to the existing LLVM representation of debug information needed to
support the :doc:`AMDGPUDwarfExtensionsForHeterogeneousDebugging`. This involved
an evaluation of the existing debug information for machine locations in LLVM,
which uncovered some hard-to-fix bugs rooted in the incidental complexity and
inconsistency of LLVM’s debug intrinsics and expressions.

Attempting to address these bugs in the existing framework proved more difficult
than expected. It became apparent that the shortcomings of the existing solution
were a direct consequence of the complexity, ambiguity, and lack of
composability encountered in DWARF.

With this in mind, we revisited the DWARF extensions to see if they could inform
a more tractable design for LLVM. We had already worked to address the
complexity and ambiguity of DWARF by defining a formalization for its expression
language and improved the composability by unifying values and location
descriptions on the evaluation stack. Together, these changes also increased the
expressiveness of DWARF. Using similar ideas in LLVM allowed us to support
additional real world cases and describe existing cases with greater accuracy.

This led us to start from the DWARF extensions and design a new set of debug
information representations. This was very heavily influenced by prior art in
LLVM, existing RFCs, mailing list discussions, review comments, and bug reports,
without which we would not have been able to make this proposal. Some of the
influences include:

-  The use of intrinsics to capture local LLVM values keeps the proposal close
   to the existing implementation, and limits the incidental work needed to
   support it for the reasons outlined in `[LLVMdev] [RFC] Separating Metadata
   from the Value hierarchy
   <https://lists.llvm.org/pipermail/llvm-dev/2014-November/078682.html>`__.
-  Support for debug locations which depend on multiple LLVM values is required
   by several optimizations, including expressing induction variables, which is
   the motivation for `D81852 [DebugInfo] Update MachineInstr interface to
   better support variadic DBG_VALUE instructions
   <https://reviews.llvm.org/D81852>`__.
-  Our solution also generalizes the notion of “fragments” to support composing
   with arbitrary expressions. For example, fragmentation can be represented
   even in the presence of arithmetic operators, as occurs in `D70601 Disallow
   DIExpressions with shift operators from being fragmented
   <https://reviews.llvm.org/D70601>`__.
-  The desire to support multiple concurrent locations for the same variable is
   described in detail in `[llvm-dev] Proposal for multi location debug info
   support in LLVM IR
   <https://lists.llvm.org/pipermail/llvm-dev/2015-December/093535.html>`__
   (continued at `[llvm-dev] Proposal for multi location debug info support in
   LLVM IR
   <https://lists.llvm.org/pipermail/llvm-dev/2016-January/093627.html>`__) and
   `Multi Location Debug Info support for LLVM
   <https://gist.github.com/Keno/480b8057df1b7c63c321>`__. Support for
   overlapping location list entries was added in DWARF 5.
-  Bugs, like `Bug 40628 - [DebugInfo@O2] Salvaged memory loads can observe
   subsequent memory writes <https://bugs.llvm.org/show_bug.cgi?id=40628>`__,
   which was partially worked around in `D57962 [DebugInfo] PR40628: Don’t
   salvage load operations <https://reviews.llvm.org/D57962>`__, often result
   from passes being unable to accurately represent the relationship between
   source variables. Our approach supports encoding that information in debug
   information in a mechanical way, with straightforward semantics.
-  Use of ``distinct`` for our new metadata nodes is motivated by use cases
   similar to those in `[LLVMdev] [RFC] Separating Metadata from the Value
   hierarchy (David Blaikie)
   <https://lists.llvm.org/pipermail/llvm-dev/2014-November/078656.html>`__
   where the content of a node is not sufficient context to unique it.

The least error prone place to make changes to debug information is at the point
where the underlying code is being transformed, hence the LLVM extensions’
representation is biased for this case.

The expression evaluation stack contains uniform pairs of location description
and type, such that all operations have well-defined semantics and no
side-effects on the evaluation of the surrounding expression. These same
semantics apply equally throughout the compiler. This allows for referentially
transparent updates, which can be reasoned about in the context of a single
operation and its inputs and outputs, rather than the space of all possible
surrounding operations and dependent expressions.

By eliminating any implicit expression inputs or operations and constraining the
state space of expressions using well-formedness rules, it is unambiguous
whether a given transformation is valid and semantics-preserving, without ever
having to consider anything outside of the expression itself.

Designing around a separation of concerns regarding expression modification and
simplification allows each update to the debug information to introduce
redundant or sub-optimal expressions. To address this, an independent
“optimizer” can simplify and canonicalize expressions. As the expression
semantics are well-defined, an“optimizer” can be run without specific knowledge
of the changes made by any one pass or combination of passes.

Incorporating a means to express “factoring”, or the definition of one
expression in terms of one or more other expressions, makes “shallow”updates
possible, bounding the work needed for any given update. This factoring is
usually trivial at the time the expression is created, but expensive to infer
later. Factored expressions can result in more compact debug information by
leveraging dynamic calling of DWARF procedures in DWARF 5, and we expect to be
able to use factoring for other purposes, such as debug information for
divergent control flow (see :ref:`amdgpu-dwarf-dw-at-llvm-lane-pc`). It is
possible to statically “flatten” this factored representation later, if required
by the debug information format being emitted, or if the emitter determines it
would be more profitable to do so.

Leveraging the DWARF extensions as a foundation, the concept of a location
description is used as the fundamental means of recording debug information. To
support this, each LLVM entity which can be referenced by an expression has a
well-defined location description, and is referred to by expressions in an
explicit, referentially transparent manner. This makes updates to reflect
changes in the underlying LLVM representation mechanical, robust, and simple.
Due to factoring, these updates are also more localized, as updates to an
expression are transparently reflected in all dependent expressions without
having to traverse them, or even be aware of their existence.

Without this factoring, any changes to an LLVM entity which are effectively used
as an input to one or more expressions would need to be“macro-expanded” at the
time they are made, in each place they are referenced. This in turn inhibits the
valid transformations the context-insensitive “optimizer” can safely perform, as
perturbing the macro-expanded expression for an LLVM entity makes it impossible
to reflect future changes to that entity in the expression. Even if this is
considered acceptable, once expressions begin to effectively depend on other
expressions (for example, in the description of induction variables, where one
program object depends on multiple other program objects) there is no longer a
bound on the recursive depth of expressions which need to be visited for any
given update, making even simple updates expensive in terms of compiler
resources. Furthermore, this approach requires either a combinatorial explosion
of expressions to describe cases when the live ranges of multiple program
objects are not equal, or the dropping of debug information for all but one such
object. None of these tradeoffs were considered acceptable.

Changes from LLVM Language Reference Manual
===========================================

This section describes a provisional set of changes to the :doc:`LangRef` to
support the :doc:`AMDGPUDwarfExtensionsForHeterogeneousDebugging`. It is not
currently fully implemented and is subject to change.

.. _amdgpu-llvm-debug-external-definitions:

External Definitions
--------------------

Some required concepts are defined outside of this document. We reproduce some
parts of those definitions, along with some expansion on their relationship to
this proposal and any extensions.

Well-Formed
~~~~~~~~~~~

The definition of “well-formed” is the one from the :ref:`LLVM Language
Reference Manual <wellformed>`.

Type
~~~~

The definition of “type” is the one from the :ref:`LLVM Language Reference
Manual <typesystem>`.

Value
~~~~~

The definition of “value” is the one from the :doc:`LangRef`.

Location Description
--------------------

The definitions of “location description”, “single location description”, and
“location storage” are the ones from the section titled
:ref:`amdgpu-dwarf-location-description` in the DWARF Extensions For
Heterogeneous Debugging.

A location description can consist of one or more single location descriptions.
A single location description specifies a location storage and bit offset. A
location storage is a linear stream of bits with a fixed size.

The storage encompasses memory, registers, and literal/implicit values.

Zero or more single location descriptions may be active for a location
description at the same instruction.

LLVM Debug Information Expressions
----------------------------------

*[Note: LLVM expressions derive much of their semantics from the DWARF
expressions described in the* :ref:`amdgpu-dwarf-expressions`\ *.]*

LLVM debug information expressions (“LLVM expressions”) specify a typed
location. *[Note: Unlike DWARF expressions, they cannot directly describe how to
compute a value. Instead, they are able to describe how to define an implicit
location description for a computed value.]*

If the evaluation of an LLVM expression does not encounter an error, then it
results in exactly one pair of location description and type.

If the evaluation of an LLVM expression encounters an error, the result is an
evaluation error.

If an LLVM expression is not well-formed, then the result is undefined.

The following sections detail the rules for when a LLVM expression is not
well-formed or results in an evaluation error.

LLVM Expression Evaluation Context
----------------------------------

An LLVM expression is evaluated in a context that includes the same context
elements as described in :ref:`amdgpu-dwarf-expression-evaluation-context` with
the following exceptions. The *current result kind* is not applicable as all
LLVM expressions are location descriptions. The *current object* and *initial
stack* are not applicable as LLVM expressions have no implicit inputs.

Location Descriptions Of LLVM Entities
--------------------------------------

The notion of location storage is extended to include the abstract LLVM entities
of *values*, *global variables*, *stack slots*, *virtual registers*, and
*physical registers*. In each case the location storage conceptually holds the
value of the corresponding entity.

For global variables, the location storage corresponds to the SSA value for the
address of the global variable as is the case when referenced in LLVM IR.

In addition, an implicit address location storage kind is defined. The size of
the storage matches the size of the type for the address. The value in the
storage is only meaningful when used in its entirety by a ``DIOpDeref``
operation, which yields a location description for the entity that the address
references. *[Note: This is a generalization to the implicit pointer location
description of DWARF 5.]*

Location descriptions can be associated with instances of any of these location
storage kinds.

High Level Structure
--------------------

Global Variable
~~~~~~~~~~~~~~~

The definition of “global variable” is the one from the :ref:`globalvars` with
the following addition.

The optional ``dbg.def`` metadata attachment can be used to specify a
``DIFragment`` termed a global variable fragment. The location description of a
global variable fragment is a memory location description for a pointer to the
global variable that references it.

If a global variable fragment is referenced by more than one global variable
``dbg.def`` field, then it is not well-formed. If a global variable fragment is
referenced by the ``object`` field of a ``DILifetime`` then it is not
well-formed.

*[Note: Global variables in LLVM exist for the duration of the program. The
global variable fragment can be referenced by the* ``argObjects`` *field of a
computed lifetime segment to specify the location for a* ``DIGlobalVariable``
*for that entire program duration. However, the global variable may exist in a
different location for a given part of the subprogram. This can be expressed
using bounded lifetime segments for the* ``DIGlobalVariable``\ *. If the
computed lifetime segment is specified, it only applies for the program
locations not covered by a bounded lifetime segment. If the computed lifetime
segment is not specified, and no bounded lifetime segment covers the program
location, then the* ``DIGlobalVariable`` *location is the undefined location
description for that program location. The bounded lifetime segments of a*
``DIGlobalVariable`` *can also reference the global variable fragment. This
allows the same LLVM global variable to be used for different*
``DIGlobalVariable``\ *s over different program locations.]*

.. TODO::

   Should there be a separate ``DIGlobalFragment`` for this since it is not
   allowed to have any bounded lifetime segments referencing it? Of should a
   ``DIFragment`` have a ``kind`` field that indicates if it is a ``computed``,
   ``bounded``, or ``global`` fragment?

..

.. TODO::

   Should the global variable fragment be the location description of the LLVM
   global variable rather than an implicit location description that is a
   pointer to it? That would void needing the ``DIOpDeref`` when referencing the
   global variable fragment. Seems can use ``DIOpAddrOf`` if need the address,
   and all other uses need the location description of the actual LLVM global
   variable. But DWARF has limitations in supporting ``DIAddrOf`` due to
   limitations in creating implicit pointer location descriptions.

Metadata
--------

An abstract metadata node exists only to abstractly specify common aspects of
derived node types, and to refer to those derived node types generally. Abstract
node types cannot be created directly.

.. _amdgpu-llvm-debug-diobject:

``DIObject``
~~~~~~~~~~~~

A ``DIObject`` is an abstract metadata node that represents the identity of a
program object used to hold data. There are several kinds of program objects.

``DIVariable``
^^^^^^^^^^^^^^

A ``DIVariable`` is a ``DIObject``, which represents the identity of a source
language program variable or non-source language program variable.

A non-source language program variable includes ``DIFlagArtificial`` in the
``flags`` field.

*[Note: A non-source language program variable may be introduced by the
compiler. These may be used in expressions needed for describing debugging
information required by the debugger.]*

*[Example: An implicit variable needed for calculating the size of a dynamically
sized array.]*

``DIGlobalVariable``
''''''''''''''''''''

A ``DIGlobalVariable`` is a ``DIVariable``, which represents the identity of a
global variable. See :ref:`DIGlobalVariable`.

``DILocalVariable``
'''''''''''''''''''

A ``DILocalVariable`` is a ``DIVariable``, which represents the identity of a
local variable. See :ref:`DILocalVariable`.

``DIFragment``
^^^^^^^^^^^^^^

.. code:: llvm

   distinct !DIFragment()

A ``DIFragment`` is a ``DIObject``, which represents the identity of a location
description that can be used as the piece of another location description.

*[Note: Unlike a* ``DIVariable``\ *, a* ``DIFragment`` *is not named and so is
not directly exposed to the user of a debugger.]*

*[Note: A* ``DIFragment`` *may be a piece of a* ``DIVariable`` *directly, or
indirectly by virtue of being a piece of some other* ``DIFragment``\ *.]*

*[Note: A* ``DIFragment`` *may be introduced to factor the definition of part of
a location description shared by other location descriptions for convenience or
to permit more compact debug information.]*

*[Note: A* ``DIFragment`` *may be introduced to allow the compiler to specify
multiple lifetime segments for the single location description referenced for a
default or type lifetime segment.]*

*[Note: In DWARF a* ``DIFragment`` *can be represented using a*
``DW_TAG_dwarf_procedure`` *DIE.]*

*[Example: The fragments into which SRoA splits a source language variable. The
location description of the source language variable would then use an
expression that combines the fragments appropriately.]*

*[Example: Divergent control flow can be described by factoring information
about how to determine active lanes by lexical scope, which results in more
compact debug information.]*

*[Note:* ``DIFragment`` *replaces using* ``DW_OP_LLVM_fragment`` *in the current
LLVM IR* ``DIExpression`` *operations. This simplifies updating expressions
which now purely describe the location description.]*

``DICode``
~~~~~~~~~~

A ``DICode`` is an abstract metadata node that represents the identity of a
program code location. There are several kinds of program code locations.

``DILabel``
^^^^^^^^^^^

A ``DILabel`` is a ``DICode``, which represents the identity of a source
language label. See :ref:`DILabel`.

``DIExprCode``
^^^^^^^^^^^^^^

.. code:: llvm

   distinct !DIExprCode()

A ``DIExprCode`` is a ``DICode``, which represents a code location that can be
referenced by the ``argObjects`` field of a ``DILifetime`` as an argument to its
``location`` field’s ``DIExpr``.

*[Note:* ``DIExprCode`` *does not represent a source language label and so
generates no debug information in itself. It is only used to allow a* ``DIExpr``
*to refer to a code location address.]*

.. _amdgpu-llvm-debug-dicompositetype:

``DICompositeType``
~~~~~~~~~~~~~~~~~~~

A ``DICompositeType`` represents the identity of a composite source program
type. See :ref:`DICompositeType`.

For ``DICompositeType`` with a ``tag`` field of ``DW_TAG_array_type``, the
optional ``dataLocation``, ``associated``, and ``rank`` fields specify a
``DIFragment`` which is termed a type property fragment.

If a type property fragment is referenced by the ``argObjects`` field of a
``DILifetime`` or by more than one ``DICompositeType`` field, then the metadata
is not well-formed.

*[Note: The* ``DILifetime``\ *(s) that reference the type property fragment
specify the location description of the type property. Their* ``location``
*field expression can use the* :ref:`amdgpu-llvm-debug-diobject` *operation to
get the location description of the instance of the composite type for which the
property is being evaluated. Their* ``argObjects`` *field can be used to specify
other* ``DIObject``\ *s if necessary.]*

``DILifetime``
~~~~~~~~~~~~~~

.. code:: llvm

   distinct !DILifetime(object: !DIObject, location: !DIExpr [, argObjects: {!DIObject,...} ] )

Represents a lifetime segment of a data object. A lifetime segment specifies a
location description expression, references a data object either explicitly or
implicitly, and defines when the lifetime segment applies. The location
description of a data object is defined by the, possibly empty, set of lifetime
segments that reference it.

.. TODO::

   Write up the fact that after LiveDebugValues this rule is amended, such that
   for a bounded lifetime segment a call to ``llvm.dbg.def``/``llvm.dbg.kill``
   is local to the basic block. That is, rather than respecting control flow
   `llvm.dbg.def`` extends either to exactly one ``llvm.dbg.def`` in the same
   basic block, or to the end of the basic block.

There are two kinds of lifetime segment:

-  A *bounded lifetime segment* is one referenced by the first argument of a
   call to the ``llvm.dbg.def`` or ``llvm.dbg.kill`` intrinsic.

   A bounded lifetime segment is termed active if the current program location’s
   instruction is in the range covered. The call to the ``llvm.dbg.def``
   intrinsic which specifies the ``DILifetime`` is the start of the range, which
   extends along all forward control flow paths until either a call to a
   ``llvm.dbg.kill`` intrinsic which specifies the same ``DILifetime``, or to
   the end of an exit basic block.

   If a bounded lifetime segment is not referenced by exactly one call ``D`` to
   the ``llvm.dbg.def`` intrinsic, then the metadata is not well-formed.

   A bounded lifetime segment can be referenced by zero or more
   ``llvm.dbg.kill`` intrinsics ``K``. If any member of ``K`` is not reachable
   from ``D`` by following control flow, or if every control flow path for every
   member of ``K`` passes through another member of ``K``, then the metadata is
   not well-formed.

   See :ref:`amdgpu-llvm-debug-llvm-dbg-def` and
   :ref:`amdgpu-llvm-debug-llvm-dbg-kill`.
-  A *computed lifetime segment* is one not referenced.

A ``DILifetime`` which does not match exactly one of the above kinds is not
well-formed.

The required ``object`` field specifies the data object of the lifetime segment.

The location description of a ``DIObject`` is a function of the current program
location’s instruction and the, possibly empty, set of lifetime segments with an
``object`` field that references the ``DIObject``:

-  If the ``DIObject`` is a global variable fragment, then the location
   description is comprised of an implicit location description that has a
   pointer value to the global variable that has a ``dbg.def`` metadata
   attachment that references it. If a global variable fragment is referenced by
   more than one global variable ``dbg.def`` metadata attachment or is
   referenced by the ``object`` field of a ``DILifetime``, then the metadata is
   not well-formed.
-  Otherwise, if the current program location is defined, and any bounded
   lifetime segment is active, then the location description is comprised of all
   of the location descriptions of all active bounded lifetime segments.
-  Otherwise, if there is a computed lifetime segment, then the location
   description is comprised of the location description of the computed lifetime
   segment. *[Note: A computed lifetime segment corresponds to the DWARF*
   ``loclist`` *default location description.]*
-  Otherwise, the location description is the undefined location description.

*[Note: When multiple bounded lifetime segments for the same*
``DIObject`` *are active at a given instruction, it describes the
situation where an object exists simultaneously in more than one place.
For example, a variable may exist in memory and then be promoted to a
register where it is only read before being clobbered and reverting to
using the memory location. While promoted to the register, a debugger
may read from either the register or memory since they both have the
same value but must update both the register and memory if the value of
the variable needs to be changed.]*

*[Note: A* ``DIObject`` *with no* ``DILifetime``\ *s has an undefined location
description. If the* ``argObjects`` *field of a* ``DILifetime`` *references such
a* ``DIObject`` *then the argument can be removed, and the* ``location``
*expression updated to use the* ``DIOpConstant`` *with an* ``undef`` *value.]*

The location description of a ``DICode`` is a single implicit location
description with a value that is the address of the start of the basic block
that contain the ``llvm.dbg.label`` intrinsic that references it. If a
``DICode`` is not referenced by exactly one call to the ``llvm.dbg.label``
intrinsic, then the metadata is not well-formed. See
:ref:`amdgpu-llvm-debug-llvm-dbg-label`.

The optional ``argObjects`` field specifies a tuple of zero or more input
``DIObject``\ s or ``DICode``\ s to the expression specified by the ``location``
field. Omitting the ``argObjects`` field is equivalent to specifying it to be
the empty tuple.

The required ``location`` field specifies the expression which evaluates to the
location description of the lifetime segment.

*[Note: The expression may refer to an argument specified by the* ``argObjects``
*field using the* :ref:`amdgpu-llvm-debug-dioparg` *operation and specifying its
zero-based position in the tuple.*

*The expression of a bounded lifetime segment may refer to the LLVM entity
specified by the second argument of the call to the* ``llvm.dbg.def`` *intrinsic
that references it using the* :ref:`amdgpu-llvm-debug-diopreferrer` *operation.*

*The expression of a lifetime segment may refer to the object instance of a type
for which a type property is being specified using the*
:ref:`amdgpu-llvm-debug-dioptypeobject` *operation.*

*The expression of a lifetime segment may refer to a global variable in LLVM by
using the* :ref:`amdgpu-llvm-debug-dioparg` *operation to refer to a global
variable fragment referenced in the* ``argObjects`` *field.]*

The reachable lifetime graph is the transitive closure of the graph formed by
the edges:

-  From each ``DIVariable`` (termed root nodes and also termed reachable
   ``DIObject``\ s) to the ``DILifetime``\ s that reference them (termed
   reachable ``DILifetime``\ s).
-  From each ``DICompositeType`` (termed root nodes) to the ``DIFragment``\ s
   that are referenced by the optional ``dataLocation``, ``associated``, and
   ``rank`` fields (termed reachable ``DIVariable``\ s).
-  From each reachable ``DILifetime`` to the ``DIObject``\ s or ``DICode``\ s
   referenced by their ``argObjects`` fields (termed reachable ``DIObject``\ s
   or reachable ``DICode``\ s respectively).
-  From each reachable ``DIObject`` to the ``DILifetime``\ s that reference them
   (termed reachable ``DILifetime``\ s).

If the reachable lifetime graph has any cycles or if any ``DILifetime``,
``DIFragment``, or ``DIExprCode`` are not in the reachable lifetime graph, then
the metadata is not well-formed.

*[Note: In current debug information the* ``DILifetime`` *information is part of
the debug intrinsics. A new lifetime for an object is defined by using a debug
intrinsic to start a new lifetime. This means an object can have at most one
active lifetime for any given program location. Separating the lifetime
information into a separate metadata node allows there to be multiple debug
intrinsics to begin different lifetime segments over the same program locations.
It also allows a debug intrinsic to indicate the end of the lifetime by
referencing the same lifetime as the intrinsic that started it.]*

``DICompileUnit``
~~~~~~~~~~~~~~~~~

A ``DICompileUnit`` represents the identity of source program compile unit. See
:ref:`DICompileUnit`.

All ``DICompileUnit`` compile units are required to be referenced by the
``!llvm.dbg.cu`` named metadata node of the LLVM module.

All ``DIGlobalVariable`` global variables of the compile unit are required to be
referenced by the ``globals`` field of the ``DICompileUnit``.

``DISubprogram``
~~~~~~~~~~~~~~~~

A ``DISubprogram`` represents the identity of source language program or
non-source language program function. See :ref:`DISubprogram`.

A non-source language program function includes ``DIFlagArtificial`` in the
``flags`` field.

All ``DILocalVariable`` local variables, ``DILabel`` labels, and ``DIExprCode``
code locations of the function are required to be referenced by the
``retainedNodes`` field of the ``DISubprogram``.

For all ``DILifetime`` computed lifetime segments that are part of the reachable
lifetime graph:

1. If only involve ``DILocalVariable``\ s, ``DICompositeType``\ s, and bounded
   lifetime segments of the same function, then are required to be referenced by
   the ``retainedNodes`` field of the corresponding ``DISubprogram``.
2. Otherwise, are required to be referenced by the ``!llvm.dbg.retainedNodes``
   named metadata node of the LLVM module.

*[Note: At the time computed lifetime segments are created, it is always well
defined if they are local to a function or are global.*

*For example, a computed lifetime segment created only to define the location of
a local variable (or a piece of a local variable), would be retained by the
function that defines the local variable. If the function were deleted there is
no need for the computed lifetime segment any more.*

*Similarly, a computed lifetime segment that contributes a lifetime to the
location description of a global variable (or fragment of a global variable)
using only local variables (or fragments of local variables) or bounded lifetime
segments of the same function, would be retained by the function that defines
the local variables (or fragments of local variables) or owns the bounded
lifetime segments. If the function were deleted there is no need for the
computed lifetime segment any more as the local variable (or fragment of a local
variable) references would need to be replaced with the undefined location
description, and the bounded lifetime segments would never be active.*

*Otherwise, the computed lifetime segment applies to a global variable (or
fragment of a global variable) and either involves other global variables (or
fragments of global variables) or local variables (or fragments of local
variables) of multiple subprograms, and therefore needs to be retained by the
LLVM module. Deleting a subprogram must not delete the computed lifetime
segment, although any references to deleted local variables (or fragments of
deleted local variables) would need to be updated to be the undefined location
description.]*

``DIExpr``
~~~~~~~~~~

.. code:: llvm

   !DIExpr(DIOp, ...)

Represents an expression, which is a sequence of one or more operations defined
in the following sections.

The evaluation of an expression is done in the context of an associated
``DILifetime`` that has a ``location`` field that references it.

The evaluation of the expression is performed on an initially empty stack where
each stack element is a tuple of a type and a location description. The
expression is evaluated by evaluating each of its operations sequentially.

The result of the evaluation is the typed location description of the single
resulting stack element. If the stack does not have a single element after
evaluation, then the expression is not well-formed.

.. TODO::

   Maybe operators should specify their input type(s)? It does not match what
   DWARF does currently. Such types cannot trivially be used to enforce type
   correctness since the expression language is an arbitrary stack, and in
   general the whole expression has to be evaluated to determine the input types
   to a given operation.

Each operation definition begins with a specification which describes the
parameters to the operation, the entries it pops from the stack, and the entries
it pushes on the stack. The specification is accepted by the modified BNF
grammar in *Figure 1—LLVM IR Expression Operation Specification Syntax*, where
``[]`` denotes character classes, ``*`` denotes zero-or-more repetitions of a
term, and ``+`` denotes one-or-more repetitions of a term.

**Figure 1—LLVM IR Expression Operation Specification Syntax**

.. code:: bnf

   <operation-specification> ::= <operation-syntax> <operation-stack-effects>

          <operation-syntax> ::= <operation-identifier> "(" <parameter-list> ")"
            <parameter-list> ::= "" | <parameter-binding-list>
    <parameter-binding-list> ::= <parameter-binding> ( ", " <parameter-binding> )+
         <parameter-binding> ::= <binding-identifier> ":" <parameter-binding-kind>
    <parameter-binding-kind> ::= "type" | "unsigned" | "literal" | "addrspace"

   <operation-stack-effects> ::= "{" <stack-list> "->" <stack-list> "}"
                <stack-list> ::= "" | <stack-binding-list>
        <stack-binding-list> ::= <stack-binding> ( " " <stack-binding> )+
             <stack-binding> ::= "(" <binding-identifier> ":" <llvm-type> ")"

      <operation-identifier> ::= [A-Za-z]+
        <binding-identifier> ::= [A-Z] [A-Z0-9]* "'"*

The ``<operation-syntax>`` describes the LLVM IR concrete syntax of the
operation in an expression.

The ``<parameter-binding-list>`` defines positional parameters to the operation.
Each parameter in the list has a ``<binding-identifier>`` which binds to the
argument passed via the parameter, and a ``<parameter-binding-kind>`` which
defines the kind of arguments accepted by the parameter.

The ``<parameter-binding-kind>`` describes the kind of the parameter:

-  ``type``: An LLVM type.
-  ``unsigned``: A non-negative literal integer.
-  ``literal``: An LLVM literal value expression.
-  ``addrspace``: An LLVM target-specific address space identifier.

The ``<operation-stack-effects>`` describe the effect of the operation on the
stack. The first ``<stack-binding-list>`` describes the “inputs”to the
operation, which are the entries it pops from the stack in the left-to-right
order. The second ``<stack-binding-list>`` describes the“outputs” of the
operation, which are the entries it pushes onto the stack in a right-to-left
order. In both cases the top stack element comes first on the left.

If evaluation can result in a stack with fewer entries than required by an
operation, then the expression is not well-formed.

Each ``<stack-binding>`` is a pair of ``<binding-identifier>`` and
``<llvm-type>``. The ``<binding-identifier>`` binds to the location description
of the stack entry. The ``<llvm-type>`` binds to the type of the stack entry and
denotes an LLVM type as defined in the :ref:`LLVM Language Reference Manual
<typesystem>`.

Each ``<binding-identifier>`` identifies a meta-syntactic variable, and each
``<llvm-type>`` may identify one or more meta-syntactic variables. When reading
the ``specification`` left-to-right, the first mention binds the meta-syntactic
variable to an entity, and subsequent mentions are an assertion that they are
the identical bound entity. If evaluation can result in parameters and stack
inputs that do not conform to the assertions, then the expression is not
well-formed. The assertions for stack outputs define post-conditions of the
operation output.

The remaining body of the definition for an operation may reference the bound
meta-syntactic variable identifiers from the specification and may define
additional meta-syntactic variables following the same left-to-right binding
semantics.

In the operation definitions, the following functions are defined:

-  ``bitsizeof(X)``: computes the size in bits of ``X``.
-  ``sizeof(X)``: computes ``bitsizeof(X) * 8``.
-  ``read(L, T)``: computes the value of type ``T`` obtained by retrieving
   ``bitsizeof(T)``: bits from location description ``L``. If any bit of the
   value retrieved is from the undefined location storage or the offset of any
   bit exceeds the size of the location storage specified by any single location
   description of ``L``, then the expression is not well-formed.

.. TODO::

   Consider defining reading undefined bits as producing an undefined location
   description. This would need DWARF to adopt this model which may be necessary
   as compilers support optimized code better. This would need all usage or
   ``read`` to be reworded to specify result if ``read`` detects undefined bits.

.. _amdgpu-llvm-debug-diopreferrer:

``DIOpReferrer``
^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpReferrer(T:type)
   { -> (L:T) }

``L`` is the location description of the referrer ``R`` of the associated
lifetime segment ``LS``. If ``LS`` is not a bounded lifetime segment, then the
expression is not well-formed.

If ``bitsizeof(T)`` is not equal to ``bitsizeof(R)``, then the expression is not
well-formed.

.. _amdgpu-llvm-debug-dioparg:

``DIOpArg``
^^^^^^^^^^^

.. code:: llvm

   DIOpArg(N:unsigned, T:type)
   { -> (L:T) }

``L`` is the location description of the ``N``\ :sup:`th` zero-based input ``I``
to the expression.

If there are fewer than ``N + 1`` inputs to the expression, then the expression
is not well-formed. If ``bitsizeof(T)`` is not equal to ``bitsizeof(I)``, then
the expression is not well-formed.

*[Note: The inputs for an expression are specified by the* ``argObjects`` *field
of the* ``DILifetime`` *being evaluated which has a* ``location`` *field that
references the expression.]*

.. _amdgpu-llvm-debug-dioptypeobject:

``DIOpTypeObject``
^^^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpTypeObject(T:type)
   { -> (L:T) }

``LS`` is the lifetime segment associated with the expression containing
``DIOpTypeObject``. ``TPF`` is the type property fragment that is evaluating
``LS``. ``LT`` is the ``DIType`` that has a type property field ``TP`` that
references ``TPF``. ``L`` is the location description of the instance ``O`` of
an object of type ``LT`` for which the type property ``TP`` is being evaluated.
See :ref:`amdgpu-llvm-debug-dicompositetype`.

If ``LS`` can be evaluated other than to obtain the location description of a
type property fragment, then the expression is not well-formed. *[Note: This
implies that a type property fragment cannot be referenced by the* ``argObjects``
*field of a* ``DILifetime``\ *.]* If ``bitsizeof(T)`` is not equal to
``bitsizeof(LT)``, then the expression is not well-formed.

.. TODO::

   Should a distinguished ``DIFragment`` be used for this like for LLVM global
   variables? There could be a uniqued type object fragment referenced by the
   ``!llvm.dbg.typeObject`` named metadata node of the LLVM module.

``DIOpConstant``
^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpConstant(T:type V:literal)
   { -> (L:T) }

``V`` is a literal value of type ``T`` or the ``undef`` value.

If ``V`` is the ``undef`` value, then ``L`` comprises one undefined location
description ``IL``.

Otherwise, ``L`` comprises one implicit location description ``IL``. ``IL``
specifies implicit location storage ``ILS`` and offset 0. ``ILS`` has value
``V`` and size ``bitsizeof(T)``.

``DIOpConvert``
^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpConvert(T':type)
   { (L:T) -> (L':T') }

``L'`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``V`` and size
``bitsizeof(T')``.

``V`` is the value ``read(L, T)`` converted to type ``T'``.

*[Note: The conversions used should be limited to those supported by the target
debug format. For example, when the target debug format is DWARF, the
conversions used should be limited to those supported by the* ``DW_OP_convert``
*operation.]*

``DIOpReinterpret``
^^^^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpReinterpret(T':type)
   { (L:T) -> (L:T') }

If ``bitsizeof(T)`` is not equal to ``bitsizeof(T')``, then the expression is
not well-formed.

``DIOpBitOffset``
^^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpBitOffset(T':type)
   { (B:I) (L:T) -> (L':T') }

``L'`` is ``L``, but updated by adding ``read(B, I)`` to its bit offset.

If ``I`` is not an integral type, then the expression is not well-defined.

*[Note:* ``I`` *may be a signed or unsigned integral type.]*

``DIOpByteOffset``
^^^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpByteOffset(T':type)
   { (B:I) (L:T) -> (L':T') }

``(L':T')`` is as if ``DIOpBitOffset(T')`` was evaluated with a stack containing
``(B * 8:I) (L:T)``.

``DIOpComposite``
^^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpComposite(N:unsigned, T:type)
   { (L1:T1) (L2:T2) ... (LN:TN) -> (L:T) }

``L`` comprises one complete composite location description ``CL`` with offset
0. The location storage associated with ``CL`` is comprised of ``N`` parts each
of bit size ``bitsizeof(TM)`` starting at the location storage specified by
``LM``. The parts are concatenated starting at offset 0 in the order with ``M``
from ``N`` to 1 and no padding between the parts.

If the sum of ``bitsizeof(TM)`` for ``M`` from 1 to ``N`` does not equal
``bitsizeof(T)``, then the expression is not well-formed.

If there are multiple parts that ultimately, after expanding referenced
composites, refer to the same bits of a non-implicit location storage, then the
expression in not well-formed.

*[Note: A debugger could not in general assign a value to such a composite
location description as different parts of the assigned value may have different
values but map to different parts of the composite location description that are
associated with same bits of a location storage. Any given bits of location
storage can only hold a single value at a time. An implicit location description
does not permit assignment, and so the same bits of its value can be present in
multiple parts of a composite location description.]*

``DIOpExtend``
^^^^^^^^^^^^^^

.. code:: llvm

   DIOpExtend(N:unsigned)
   { (L:T) -> (L':<N x T>) }

``(L':<N x T>)'`` is as if ``DIOpComposite(N, <N x T>)`` was applied to a stack
containing ``N`` copies of ``(L:T)``.

If ``T`` is not an integral type, floating point type, or pointer type, then the
expression is not well-formed.

``DIOpSelect``
^^^^^^^^^^^^^^

.. code:: llvm

   DIOpSelect()
   { (LM:TM) (L1:<N x T>) (L0:<N x T>) -> (L:<N x T>) }

``M`` is a bit mask with the value ``read(LM, TM)``. If ``bitsizeof(TM)`` is
less than ``N``, then the expression is not well-formed.

``(L:<N x T>)`` is as if ``DIOpComposite(N, <N x T>)`` was applied to a stack
containing ``N`` entries ``(LI:T)`` ordered in descending ``I`` from ``N - 1``
to 0 inclusive. Each ``LI`` is as if ``DIOpBitOffset(T)`` was applied to a stack
containing ``(I * bitsizeof(T):TI) (PLI:T)``. ``PLI`` is the same as ``L0`` if
the ``I``\ :sup:`th` least significant bit of ``M`` is zero, otherwise it is the
same as ``L1``. ``TI`` is some integral type that can represent the range 0 to
``(N - 1) * bitsizeof(T)``.

If ``T`` is not an integral type, floating point type, or pointer type, then the
expression is not well-formed.

.. _amdgpu-llvm-debug-diopaddrof:

``DIOpAddrOf``
^^^^^^^^^^^^^^

.. code:: llvm

   DIOpAddrOf(N:addrspace)
   { (L:T) -> (L':ptr addrspace(N)) }

``L'`` comprises one implicit address location description ``IAL``. ``IAL``
specifies implicit address location storage ``IALS`` and offset 0.

``IALS`` is ``bitsizeof(ptr addrspace(N))`` bits and conceptually holds a
reference to the storage that ``L`` denotes. If ``DIOpDeref(T)`` is applied to
the resulting ``(L':ptr addrspace(N))``, then it will result in ``(L:T)``. If
any other operation is applied, then the expression is not well-formed.

*[Note:* ``DIOpAddrOf`` *can be used for any location description kind of*
``L``\ *, not just memory location descriptions.]*

*[Note: DWARF only supports creating implicit pointer location descriptors for
variables or DWARF procedures. It does not support creating them for an
arbitrary location description expression. The examples below cover the current
LLVM optimizations and only use* ``DIOpAddrOf`` *applied to* ``DIOpReferrer``\
*,* ``DIOPArg``\ *, and* ``DIOpConstant``\ *. All these cases can map onto
existing DWARF in a straightforward manner. There would be more complexity if*
``DIOpAddrOf`` *was used in other situations. Such usage could either be
addressed by dropping debug information as LLVM currently does in numerous
situations, or by adding additional DWARF extensions.]*

``DIOpDeref``
^^^^^^^^^^^^^

.. code:: llvm

   DIOpDeref(T:type)
   { (L:ptr addrspace(N)) -> (L':T) }

If ``(L:ptr addrspace(N))`` was produced by a ``DIOpAddrOf`` operation, then
see :ref:`amdgpu-llvm-debug-diopaddrof`:.

Otherwise, ``L'`` comprises one memory location description ``MLD``. ``MLD``
specifies bit offset ``read(L, ptr addrspace(N)) * 8`` and the memory location
storage corresponding to address space ``N``.

``DIOpRead``
^^^^^^^^^^^^

.. code:: llvm

   DIOpRead()
   { (L:T) -> (L':T) }

``L'`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``read(L, T)``
and size ``bitsizeof(T)``.

``DIOpAdd``
^^^^^^^^^^^

.. code:: llvm

   DIOpAdd()
   { (L1:T) (L2:T) -> (L:T) }

``L`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``read(L1, T)
+ read(L2, T)`` and size ``bitsizeof(T)``.

``DIOpSub``
^^^^^^^^^^^

.. code:: llvm

   DIOpSub()
   { (L1:T) (L2:T) -> (L:T) }

``L`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``read(V2, T)
- read(V1, T)`` and size ``bitsizeof(T)``.

``DIOpMul``
^^^^^^^^^^^

.. code:: llvm

   DIOpMul()
   { (L1:T) (L2:T) -> (L:T) }

``L`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``read(V2, T)
* read(V1, T)`` and size ``bitsizeof(T)``.

``DIOpDiv``
^^^^^^^^^^^

.. code:: llvm

   DIOpDiv()
   { (L1:T) (L2:T) -> (L:T) }

``L`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``read(V2, T)
/ read(V1, T)`` and size ``bitsizeof(T)``.

``DIOpShr``
^^^^^^^^^^^

.. code:: llvm

   DIOpShr()
   { (L1:T) (L2:T) -> (L:T) }

``L`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``read(V2, T)
>> read(V1, t)`` and size ``bitsizeof(T)``. If ``T`` is an unsigned integral
type, then the result is filled with 0 bits. If ``T`` is a signed integral type,
then the result is filled with the sign bit of ``V1``.

If ``T`` is not an integral type, then the expression is not well-formed.

``DIOpShl``
^^^^^^^^^^^

.. code:: llvm

   DIOpShl()
   { (L1:T) (L2:T) -> (L:T) }

``L`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has value ``read(V2, T)
<< read(V1, T)`` and size ``bitsizeof(T)``. The result is filled with 0 bits.

If ``T`` is not an integral type, then the expression is not well-formed.

``DIOpPushLane``
^^^^^^^^^^^^^^^^

.. code:: llvm

   DIOpPushLane(T:type)
   { -> (L:T) }

``L`` comprises one implicit location description ``IL``. ``IL`` specifies
implicit location storage ``ILS`` and offset 0. ``ILS`` has the value of the
target architecture lane identifier of the current source language thread of
execution if the source language is implemented using a SIMD or SIMT execution
model.

If ``T`` is not an integral type or the source language is not implemented using
a SIMD or SIMT execution model, then the expression is not well-formed.

Intrinsics
----------

The intrinsics define the program location range over which the location
description specified by a bounded lifetime segment of a ``DILifetime`` is
active. They support defining a single or multiple locations for a source
program variable. Multiple locations can be active at the same program location
as supported by :ref:`amdgpu-dwarf-location-list-expressions`.

.. _amdgpu-llvm-debug-llvm-dbg-def:

``llvm.dbg.def``
~~~~~~~~~~~~~~~~

.. code:: llvm

   void @llvm.dbg.def(metadata, metadata)

The first argument to ``llvm.dbg.def`` is required to be a ``DILifetime`` and is
the beginning of the bounded lifetime being defined.

The second argument to ``llvm.dbg.def`` is required to be a value-as-metadata
and defines the LLVM entity acting as the referrer of the bounded lifetime
segment specified by the first argument. A value of ``undef`` is allowed and
specifies the undefined location description.

*[Note:* ``undef`` *can be used when the lifetime segment expression does not
use a* ``DIOpReferrer`` *operation, either because the expression evaluates to a
constant implicit location description, or because it only uses* ``DIOpArg``
*operations for inputs.]*

The MC pseudo instruction equivalent is ``DBG_DEF`` which has the same two
arguments with the same meaning:

.. code:: llvm

   DBG_DEF metadata, <value>

.. _amdgpu-llvm-debug-llvm-dbg-kill:

``llvm.dbg.kill``
~~~~~~~~~~~~~~~~~

.. code:: llvm

   void @llvm.dbg.kill(metadata)

The argument to ``llvm.dbg.kill`` is required to be a ``DILifetime`` and is the
end of the lifetime being killed.

Every call to the ``llvm.dbg.kill`` intrinsic is required to be reachable from a
call to the ``llvm.dbg.def`` intrinsic which specifies the same ``DILifetime``,
otherwise it is not well-formed.

The MC pseudo instruction equivalent is ``DBG_KILL`` which has the same argument
with the same meaning:

.. code:: llvm

   DBG_KILL metadata

.. _amdgpu-llvm-debug-llvm-dbg-label:

``llvm.dbg.label``
~~~~~~~~~~~~~~~~~~

.. code:: llvm

   void @llvm.dbg.label(metadata)

The argument to ``llvm.dbg.label`` is required to be a ``DICode`` and defines
its address value to be the code address of the start of the basic block that
contains it.

The MC pseudo instruction equivalent is ``DBG_LABEL`` which has the same
argument with the same meaning:

.. code:: llvm

   DBG_LABEL metadata

Examples
========

Examples which need meta-syntactic variables prefix them with a sigil to
concisely give context. The prefix sigils are:

========= ========================================================
**Sigil** **Meaning**
========= ========================================================
%         SSA IR Value
$         Non-SSA MIR Register (for example, post phi-elimination)
#         Arbitrary literal constant
========= ========================================================

The syntax used in the examples attempts to match LLVM IR/MIR as closely as
possible, with the only new syntax required being that of the expression
language.

Variable Located In An ``alloca``
---------------------------------

The frontend will generate ``alloca``\ s for every variable, and can trivially
insert a single ``DILifetime`` covering the whole body of the function, with the
expression ``DIExpr(DIOpReferrer(<type>*), DIOpDeref(<type>)``, referring to the
``alloca``. Walking the debug intrinsics provides the necessary information to
generate the DWARF ``DW_AT_location`` attributes on variables.

.. code:: llvm
   :number-lines:

   %x.addr = alloca i64, addrspace(5)
   call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x.addr)
   store i64* %x.addr, ...
   ...
   call void @llvm.dbg.kill(metadata !2)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref(i64)))

Variable Promoted To An SSA Register
------------------------------------

The promotion semantically removes one level of indirection, and correspondingly
in the debug expressions for which the ``alloca`` being replaced was the
referrer, an additional ``DIOpAddrOf(N)`` is needed.

An example is ``mem2reg`` where an ``alloca`` can be replaced with an SSA value:

.. code:: llvm
   :number-lines:

   %x = i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 %x)
   ...
   call void @llvm.dbg.kill(metadata !2)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5), DIOpDeref(i64)))

The canonical form of this is then just ``DIOpReferrer(i64)`` as the pair of
``DIOpAddrOf(N)``, ``DIOpDeref(i64)`` cancel out:

.. code:: llvm
   :number-lines:

   %x = i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 %x)
   ...
   call void @llvm.dbg.kill(metadata !2)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))

Implicit Pointer Location Description
-------------------------------------

The transformation for removing a level of indirection is to add an
``DIOpAddrOf(N)``, which may result in a location description for a pointer to a
non-memory object.

.. code:: c
   :number-lines:

   int x = ...;
   int *p = &x;
   return *p;

.. code:: llvm
   :number-lines:

   %x.addr = alloca i64, addrspace(5)
   call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x.addr)
   store i64 addrspace(5)* %x.addr, i64 ...
   %p.addr = alloca i64*, addrspace(5)
   call void @llvm.dbg.def(metadata !4, metadata i64 addrspace(5)* addrspace(5)* %p.addr)
   store i64 addrspace(5)* addrspace(5)* %p.addr, i64 addrspace(5)* %x.addr
   %0 = load i64 addrspace(5)* addrspace(5)* %p.addr
   %1 = load i64 addrspace(5)* %0
   ret i64 %1

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref(i64)))
   !3 = !DILocalVariable("p", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64 addrspace(5)* addrspace(5)*), DIOpDeref(i64 addrspace(5)*)))

*[Note: The* ``llvm.dbg.def`` *could either be placed after the* ``alloca`` *or
after the* ``store`` *that defines the variables initial value. The difference
is whether the debugger will be able to allow the user to access the variable
before it is initialized. Proposals exist to allow the compiler to communicate
when a variable is uninitialized separately from defining its location.]*

First round of ``mem2reg`` promotes ``%p.addr`` to an SSA register ``%p``:

.. code:: llvm
   :number-lines:

   %x.addr = alloca i64, addrspace(5)
   store i64 addrspace(5)* %x.addr, i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x.addr)
   %p = i64 addrspace(5)* %x.addr
   call void @llvm.dbg.def(metadata !4, metadata i64 addrspace(5)* %p)
   %0 = load i64 addrspace(5)* %p
   return i64 %0

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref(i64)))
   !3 = !DILocalVariable("p", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpAddrOf(5), DIOpDeref(i64 addrspace(5)*)))

Simplify by eliminating ``%p`` and directly using ``%x.addr``:

.. code:: llvm
   :number-lines:

   %x.addr = alloca i64, addrspace(5)
   store i64 addrspace(5)* %x.addr, i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x.addr)
   call void @llvm.dbg.def(metadata !4, metadata i64 addrspace(5)* %x.addr)
   load i64 %0, i64 addrspace(5)* %x.addr
   return i64 %0

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*), DIOpDeref(i64)))
   !3 = !DILocalVariable("p", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*)))

Second round of ``mem2reg`` promotes ``%x.addr`` to an SSA register ``%x``:

.. code:: llvm
   :number-lines:

   %x = i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 %x)
   call void @llvm.dbg.def(metadata !4, metadata i64 %x)
   %0 = i64 %x
   return i64 %0

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5), DIOpDeref(i64)))
   !3 = !DILocalVariable("p", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5)))

Simplify by eliminating adjacent ``DIOpAddrOf(5), DIOpDeref(i64)`` and use
``%x`` directly in the ``return``:

.. code:: llvm
   :number-lines:

   %x = i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 %x)
   call void @llvm.dbg.def(metadata !2, metadata i64 %x)
   return i64 %x

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
   !3 = !DILocalVariable("p", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64), DIOpAddrOf(5)))

If ``%x`` was being assigned a constant, then can eliminated ``%x`` entirely and
substitute all uses with the constant:

.. code:: llvm
   :number-lines:

   call void @llvm.dbg.def(metadata !2, metadata i1 undef)
   call void @llvm.dbg.def(metadata !4, metadata i1 undef)
   return i64 ...

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpConstant(i64 ...)))
   !3 = !DILocalVariable("p", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpConstant(i64 ...), DIOpAddrOf(5)))

Local Variable Broken Into Two Scalars
--------------------------------------

When a transformation decomposes one location into multiple distinct ones, it
needs to follow all ``llvm.dbg.def`` intrinsics to the ``DILifetime``\ s
referencing the original location and update the expression and positional
arguments such that:

-  All instances of ``DIOpReferrer()`` in the original expression are replaced
   with the appropriate composition of all the new location pieces, now encoded
   via multiple ``DIOpArg()`` operations referring to input ``DIObject``\ s, and
   a ``DIOpComposite`` operation. This makes the associated ``DILifetime`` a
   computed lifetime segment.
-  Those location pieces are represented by new ``DIFragment``\ s, one per new
   location, each with appropriate ``DILifetime``\ s referenced by new
   ``llvm.dbg.def`` and ``llvm.dbg.kill`` intrinsics.

It is assumed that any transformation capable of doing the decomposition in the
first place needs to have all of this information available, and the structure
of the new intrinsics and metadata avoids any costly operations during
transformations. This update is also “shallow”, in that only the ``DILifetime``
which is immediately referenced by the relevant ``llvm.dbg.def``\ s need to be
updated, as the result is referentially transparent to any other dependent
``DILifetime``\ s.

.. code:: llvm
   :number-lines:

   %x = ...
   call void @llvm.dbg.def(metadata !2, metadata i64 addrspace(5)* %x)
   ...
   call void @llvm.dbg.kill(metadata !2)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64 addrspace(5)*)))

Transformed a ``i64`` SSA value into two ``i32`` SSA values:

.. code:: llvm
   :number-lines:

   %x.lo = i32 ...
   call void @llvm.dbg.def(metadata !4, metadata i32 %x.lo)
   ...
   %x.hi = i32 ...
   call void @llvm.dbg.def(metadata !6, metadata i32 %x.hi)
   ...
   call void @llvm.dbg.kill(metadata !6)
   call void @llvm.dbg.kill(metadata !4)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(1, i32), DIOpArg(0, i32), DIOpComposite(2, i64)), argObjects: {!3, !5})
   !3 = distinct !DIFragment()
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i32)))
   !5 = distinct !DIFragment()
   !6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpReferrer(i32)))

Further Decomposition Of An Already SRoA’d Local Variable
---------------------------------------------------------

An example to demonstrate the “shallow update” property is to take the above IR:

.. code:: llvm
   :number-lines:

   %x.lo = i32 ...
   call void @llvm.dbg.def(metadata !4, metadata i32 %x.lo)
   ...
   %x.hi = i32 ...
   call void @llvm.dbg.def(metadata !6, metadata i32 %x.hi)
   ...
   call void @llvm.dbg.kill(metadata !6)
   call void @llvm.dbg.kill(metadata !4)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(1, i32), DIOpArg(0, i32), DIOpComposite(2, i64)), argObjects: {!3, !5})
   !3 = distinct !DIFragment()
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i32)))
   !5 = distinct !DIFragment()
   !6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpReferrer(i32)))

and subdivide ``%x.hi`` again:

.. code:: llvm
   :number-lines:

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
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(1, i32), DIOpArg(0, i32), DIOpComposite(2, i64)), argObjects: {!3, !5})
   !3 = distinct !DIFragment()
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i32)))
   !5 = distinct !DIFragment()
   !6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpArg(1, i16), DIOpArg(0, i16), DIOpComposite(2, i32)), argObjects: {!7, !9})
   !7 = distinct !DIFragment()
   !8 = distinct !DILifetime(object: !7, location: !DIExpr(DIOpReferrer(i16)))
   !9 = distinct !DIFragment()
   !10 = distinct !DILifetime(object: !9, location: !DIExpr(DIOpReferrer(i16)))

Note that the expression for the original source variable ``x`` did not need to
be changed, as it is defined in terms of the ``DIFragment``, the identity of
which is not changed after it is created.

Multiple Live Ranges For A Single Variable
------------------------------------------

Once out of SSA, or even while in SSA via memory, there may be multiple re-uses
of the same storage for completely disparate variables, and disjoint and/or
overlapping lifetimes for any single variable. This is modeled naturally by
maintaining *defs* and *kills* for these live ranges independently at, for
example, definitions and clobbers.

.. code:: llvm
   :number-lines:

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
   RETURN

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i32)))
   !3 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i32)))
   !4 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i32)))
   !5 = !DILocalVariable("y", ...)
   !6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpReferrer(i32)))

In this example, ``$r0`` is referred to by disjoint ``DILifetime``\ s for
different variables. There is also a point where multiple ``DILifetime``\ s for
the same variable are live.

The first point implies the need for intrinsics/pseudo-instructions to define
the live range, as simply referring to an LLVM entity does not provide enough
information to reconstruct the live range.

The second point is needed to accurately represent cases where, for example, a
variable lives in both a register and in memory. The current
intrinsics/pseudo-instructions do not have the notion of live ranges for source
variables, and simply throw away at least one of the true lifetimes in these
cases.

Global Variable Broken Into Two Scalars
---------------------------------------

.. code:: llvm
   :number-lines:

   @g = i64 !dbg.def !2

   !llvm.dbg.cu = !{!0}
   !llvm.dbg.retainedNodes = !{!3}
   !0 = !DICompileUnit(..., globals: !{!1})
   !1 = !DIGlobalVariable("g")
   !2 = distinct DIFragment()
   !3 = distinct !DILifetime(
          object: !1,
          location: !DIExpr(
            DIOpArg(0, i64 addrspace(1)*),
            DIDeref()
          ),
          argObjects: {!2}
        )

Becomes:

.. code:: llvm
   :number-lines:

   @g.lo = i32 !dbg.def !2
   @g.hi = i32 !dbg.def !3

   !llvm.dbg.cu = !{!0}
   !llvm.dbg.retainedNodes = !{!4}
   !0 = !DICompileUnit(..., globals: !{!1})
   !1 = !DIGlobalVariable("g")
   !2 = distinct !DIFragment()
   !3 = distinct !DIFragment()
   !4 = distinct !DILifetime(
          object: !1,
          location: !DIExpr(
            DIOpArg(1, i32 addrspace(1)*),
            DIDeref(),
            DIOpArg(0, i32 addrspace(1)*),
            DIDeref(),
            DIOpComposite(2, i64)
          ),
          argObjects: {!2, !3}
        )

A function can specify the location of the global variable ``!1`` over some
range by simply defining bounded lifetime segments that also reference ``!1``.
These will override the “default” location description specified by the computed
lifetime segment ``!4``.

Induction Variable
------------------

Starting with some program:

.. code:: llvm
   :number-lines:

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
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
   !3 = !DILocalVariable("y", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64)))
   !5 = !DILocalVariable("i", ...)
   !6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpReferrer(i64)))

If analysis proves ``i`` over some range is equal to ``x + y``, the storage for
``i`` can be eliminated, and it can be materialized at every use. The
corresponding change needed in the debug information is:

.. code:: llvm
   :number-lines:

   %x = i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 %x)
   ...
   %y = i64 ...
   call void @llvm.dbg.def(metadata !4, metadata i64 %y)
   ...
   call void @llvm.dbg.def(metadata !6, metadata i64 undef)
   ...
   call void @llvm.dbg.kill(metadata !6)
   call void @llvm.dbg.kill(metadata !4)
   call void @llvm.dbg.kill(metadata !2)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))
   !3 = !DILocalVariable("y", ...)
   !4 = distinct !DILifetime(object: !3, location: !DIExpr(DIOpReferrer(i64)))
   !5 = !DILocalVariable("i", ...)
   !6 = distinct !DILifetime(object: !5, location: !DIExpr(DIOpArg(0, i64), DIOpArg(1, i64), DIOpAdd()), DIOpArg(!1, !3})

For the given range, the value of ``i`` is computable so long as both ``x`` and
``y`` are live, the determination of which is left until the backend debug
information generation (for example, for old DWARF or for other debug
information formats), or until debugger runtime when the expression is evaluated
(for example, for DWARF with ``DW_OP_call`` and ``DW_TAG_dwarf_procedure``).
During compilation, this representation allows all updates to maintain the debug
information efficiently by making updates “shallow”.

In other cases, this can allow the debugger to provide locations for part of a
source variable, even when other parts are not available. This may be the case
if a ``struct`` with many fields is broken up during SRoA and the lifetimes of
each piece diverge.

Proven Constant
---------------

As a very similar example to the above induction variable case (in terms of the
updates needed in the debug information), the case where a variable is proven to
be a statically known constant over some range turns the following:

.. code:: llvm
   :number-lines:

   %x = i64 ...
   call void @llvm.dbg.def(metadata !2, metadata i64 %x)
   ...
   call void @llvm.dbg.kill(metadata !2)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i64)))

into:

.. code:: llvm
   :number-lines:

   call void @llvm.dbg.def(metadata !2, metadata i64 undef)
   ...
   call void @llvm.dbg.kill(metadata !2)

   !1 = !DILocalVariable("x", ...)
   !2 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpConstant(i64 ...)))

Common Subexpression Elimination (CSE)
--------------------------------------

This is the example from `Bug 40628 - [DebugInfo@O2] Salvaged memory loads can
observe subsequent memory writes
<https://bugs.llvm.org/show_bug.cgi?id=40628>`__:

.. code:: c
   :number-lines:

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

Which after ``SROA+mem2reg`` becomes (where ``redundant`` is ``!17`` and
``loaded`` is ``!16``):

.. code:: llvm
   :number-lines:

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

And previously led to this after ``EarlyCSE``, which removes the redundant load
from ``%bar``:

.. code:: llvm
   :number-lines:

   define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !7 {
   entry:
     call void @llvm.dbg.value(metadata i32* %bar, metadata !13, metadata !DIExpression()), !dbg !18
     call void @llvm.dbg.value(metadata i32 %arg, metadata !14, metadata !DIExpression()), !dbg !18
     call void @llvm.dbg.value(metadata i32 %more, metadata !15, metadata !DIExpression()), !dbg !18

     ; This is not accurate to begin with, as a debugger which modifies
     ; `redundant` will erroneously update the pointee of the parameter `bar`.
     call void @llvm.dbg.value(metadata i32* %bar, metadata !16, metadata !DIExpression(DW_OP_deref)), !dbg !18

     %0 = load i32, i32* %bar, align 4, !dbg !19, !tbaa !20
     call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !DIExpression()), !dbg !18
     %add = add nsw i32 %more, %0, !dbg !24
     call void @llvm.dbg.value(metadata i32 undef, metadata !14, metadata !DIExpression()), !dbg !18

     ; This store "clobbers" the debug location description for `redundant`, such
     ; that a debugger about to execute the following `ret` will erroneously
     ; report `redundant` as equal to `0` when the source semantics have it still
     ; equal to the value pointed to by `bar` on entry.
     store i32 0, i32* %bar, align 4, !dbg !25, !tbaa !20
     ret i32 %more, !dbg !26
   }

But now becomes (conservatively):

.. code:: llvm
   :number-lines:

   define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !7 {
   entry:
     call void @llvm.dbg.value(metadata i32* %bar, metadata !13, metadata !DIExpression()), !dbg !18
     call void @llvm.dbg.value(metadata i32 %arg, metadata !14, metadata !DIExpression()), !dbg !18
     call void @llvm.dbg.value(metadata i32 %more, metadata !15, metadata !DIExpression()), !dbg !18

     ; The above mentioned patch for PR40628 adds special treatment, dropping
     ; the debug information for `redundant` completely in this case, making
     ; this conservatively correct.
     call void @llvm.dbg.value(metadata i32 undef, metadata !16, metadata !DIExpression()), !dbg !18

     %0 = load i32, i32* %bar, align 4, !dbg !19, !tbaa !20
     call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !DIExpression()), !dbg !18
     %add = add nsw i32 %more, %0, !dbg !24
     call void @llvm.dbg.value(metadata i32 undef, metadata !14, metadata !DIExpression()), !dbg !18
     store i32 0, i32* %bar, align 4, !dbg !25, !tbaa !20
     ret i32 %more, !dbg !26
   }

Effectively at the point of the CSE eliminating the load, it conservatively
marks the source variable ``redundant`` as optimized out.

It seems like the semantics that CSE really wants to encode in the debug
intrinsics is that, after the point at which the common load occurs, the
location for both ``redundant`` and ``loaded`` is ``%0``, and that they are both
read-only. It seems like it needs to prove this to combine them, and if it can
only combine them over some range, it can insert additional live ranges to
describe their separate locations outside of that range. The implicit pointer
example further suggests why this may need to be the case, because at the time
the implicit pointer is created, it is not known which source variable to bind
to in order to get the multiple lifetimes in this design.

This seems to be supported by the fact that even in current LLVM trunk, with the
more conservative change to mark the ``redundant`` variable as ``undef`` in the
above case, changing the source to modify ``redundant`` after the load results
in both ``redundant`` and ``loaded`` referring to the same location, and both
being read-write. A modification of ``redundant`` in the debugger before the use
of ``loaded`` is permitted and would have the effect of also updating
``loaded``. An example of the modified source needed to cause this is:

.. code:: c
   :number-lines:

   int
   foo(int *bar, int arg, int more)
   {
     int redundant = *bar;
     int loaded = *bar;
     arg &= more + loaded; // A store to redundant here affects loaded.

     *bar = redundant; // The use and subsequent modification of `redundant` here
     redundant = 1;    // effectively circumvents the patch for PR40628.

     return more + *bar;
   }

   int
   main() {
     int lala = 987654;
     return foo(&lala, 1, 2);
   }

Note that after ``EarlyCSE``, this example produces the same location
description for both ``redundant`` and ``loaded`` (metadata ``!17`` and
``!18``):

.. code:: llvm
   :number-lines:

   define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !8 {
   entry:
     call void @llvm.dbg.value(metadata i32* %bar, metadata !14, metadata !DIExpression()), !dbg !19
     call void @llvm.dbg.value(metadata i32 %arg, metadata !15, metadata !DIExpression()), !dbg !19
     call void @llvm.dbg.value(metadata i32 %more, metadata !16, metadata !DIExpression()), !dbg !19
     %0 = load i32, i32* %bar, align 4, !dbg !20, !tbaa !21

     ; The same location is reused for both source variables, without it being
     ; marked read-only (namely without it being made into an implicit location
     ; description).
     call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !DIExpression()), !dbg !19
     call void @llvm.dbg.value(metadata i32 %0, metadata !18, metadata !DIExpression()), !dbg !19

     ; Modifications to either source variable in a debugger affect the other from
     ; this point on in the function.
     %add = add nsw i32 %more, %0, !dbg !25
     call void @llvm.dbg.value(metadata i32 undef, metadata !15, metadata !DIExpression()), !dbg !19
     call void @llvm.dbg.value(metadata i32 1, metadata !17, metadata !DIExpression()), !dbg !19
     ret i32 %add, !dbg !26
   }

*[Note: To see this result, i386 is required; x86_64 seems to do even more
optimization which eliminates both* ``loaded`` *and* ``redundant``\ *.]*

Fixing this issue in the current debug information is technically possible, but
as noted by the LLVM community in the review for the attempted conservative
patch:

   *“this isn’t something that can be fixed without a lot of work, thus it’s
   safer to turn off for now.”*

The LLVM extensions make this case tractable to support with full generality and
composability with other optimizations. The expected result of ``EarlyCSE``
would be:

.. code:: llvm
   :number-lines:

   define dso_local i32 @foo(i32* %bar, i32 %arg, i32 %more) #0 !dbg !8 {
   entry:
     call void @llvm.dbg.def(metadata i32* %bar, metadata !19), !dbg !19
     call void @llvm.dbg.def(metadata i32 %arg, metadata !20), !dbg !19
     call void @llvm.dbg.def(metadata i32 %more, metadata !21), !dbg !19
     %0 = load i32, i32* %bar, align 4, !dbg !20, !tbaa !21

     call void @llvm.dbg.def(metadata i32 %0, metadata !22), !dbg !19
     call void @llvm.dbg.def(metadata i32 %0, metadata !23), !dbg !19

     %add = add nsw i32 %more, %0, !dbg !25
     ret i32 %add, !dbg !26
   }

   !14 = !DILocalVariable("bar", ...)
   !15 = !DILocalVariable("arg", ...)
   !16 = !DILocalVariable("more", ...)
   !17 = !DILocalVariable("redundant", ...)
   !18 = !DILocalVariable("loaded", ...)
   !19 = distinct !DILifetime(object: !14, location: !DIExpr(DIOpReferrer(i32*)))
   !20 = distinct !DILifetime(object: !15, location: !DIExpr(DIOpReferrer(i32)))
   !21 = distinct !DILifetime(object: !16, location: !DIExpr(DIOpReferrer(i32)))
   !21 = distinct !DILifetime(object: !17, location: !DIExpr(DIOpReferrer(i32), DIOpRead()))
   !22 = distinct !DILifetime(object: !18, location: !DIExpr(DIOpReferrer(i32), DIOpRead()))

Which accurately describes that both ``redundant`` and ``loaded`` are read-only
after the common load.

Divergent Lane PC
-----------------

For AMDGPU, the ``DW_AT_LLVM_lane_pc`` attribute is used to specify the program
location of the separate lanes of a SIMT thread.

If the lane is an active lane, then this will be the same as the current program
location.

If the lane is inactive, but was active on entry to the subprogram, then this is
the program location in the subprogram at which execution of the lane is
conceptual positioned.

If the lane was not active on entry to the subprogram, then this will be the
undefined location. A client debugger can check if the lane is part of a valid
work-group by checking that the lane is in the range of the associated
work-group within the grid, accounting for partial work-groups. If it is not,
then the debugger can omit any information for the lane. Otherwise, the debugger
may repeatedly unwind the stack and inspect the ``DW_AT_LLVM_lane_pc`` of the
calling subprogram until it finds a non-undefined location. Conceptually the
lane only has the call frames that it has a non-undefined
``DW_AT_LLVM_lane_pc``.

The following example illustrates how the AMDGPU backend can generate a DWARF
location list expression for the nested ``IF/THEN/ELSE`` structures of the
following subprogram pseudo code for a target with 64 lanes per wavefront.

.. code:: llvm
   :number-lines:

   SUBPROGRAM X
   BEGIN
     a;
     IF (c1) THEN
       b;
       IF (c2) THEN
         c;
       ELSE
         d;
       ENDIF
       e;
     ELSE
       f;
     ENDIF
     g;
   END

The AMDGPU backend may generate the following pseudo LLVM MIR to manipulate the
execution mask (``EXEC``) to linearize the control flow. The condition is
evaluated to make a mask of the lanes for which the condition evaluates to true.
First the ``THEN`` region is executed by setting the ``EXEC`` mask to the
logical ``AND`` of the current ``EXEC`` mask with the condition mask. Then the
``ELSE`` region is executed by negating the ``EXEC`` mask and logical ``AND`` of
the saved ``EXEC`` mask at the start of the region. After the ``IF/THEN/ELSE``
region the ``EXEC`` mask is restored to the value it had at the beginning of the
region. This is shown below. Other approaches are possible, but the basic
concept is the same.

.. code:: llvm
   :number-lines:

   %lex_start:
     a;
     %1 = EXEC
     %2 = c1
   %lex_1_start:
     EXEC = %1 & %2
   $if_1_then:
       b;
       %3 = EXEC
       %4 = c2
   %lex_1_1_start:
       EXEC = %3 & %4
   %lex_1_1_then:
         c;
       EXEC = ~EXEC & %3
   %lex_1_1_else:
         d;
       EXEC = %3
   %lex_1_1_end:
       e;
     EXEC = ~EXEC & %1
   %lex_1_else:
       f;
     EXEC = %1
   %lex_1_end:
     g;
   %lex_end:

To create the DWARF location list expression that defines the location
description of a vector of lane program locations, the LLVM MIR ``DBG_DEF``
pseudo instruction can be used to annotate the linearized control flow. This can
be done by defining a ``DIFragment`` for the lane PC and using it as the
``activeLanePC`` parameter of the corresponding ``DISubprogram`` of the function
being described. The DWARF location list expression created for it is used as
the value of the ``DW_AT_LLVM_lane_pc`` attribute on the subprogram’s debugger
information entry.

A ``DIFragment`` is defined for each well nested structured control flow region
which provides the conceptual lane program location for a lane if it is not
active (namely it is divergent). The ``DIFragment`` for each region has a single
computed ``DILifetime`` whose location expression conceptually inherits the
value of the immediately enclosing region and modifies it according to the
semantics of the region.

By having a separate ``DIFragment`` for each region, they can be reused to
define the value for any nested region. This reduces the total size of the DWARF
operation expressions.

A “bounded divergent lane PC” ``DIFragment`` is defined which computes the
program location for each lane assuming they are divergent at every instruction
in the function. This fragment has one bounded lifetime for each region. Each
bounded lifetime specifies a single ``DIFragment`` for a region and is active
over a disjoint range of the function instructions corresponding to that region.
Together the lifetimes cover all instructions of the function, such that at
every PC in the function exactly one lifetime is active.

For an ``IF/THEN/ELSE`` region, the divergent program location is at the start
of the region for the ``THEN`` region since it is executed first. For the
``ELSE`` region, the divergent program location is at the end of the
``IF/THEN/ELSE`` region since the ``THEN`` region has completed.

The lane PC fragment is then defined with an expression that takes the bounded
divergent lane PC and modifies it by inserting the current program location for
each lane that the ``EXEC`` mask indicates is active.

The following provides an example using pseudo LLVM MIR.

.. code:: llvm
   :number-lines:

   ; NOTE: This listing is written in a pseudo LLVM MIR, as this debug information
   ; will be inserted as part of inserting EXEC manipulation into LLVM MIR.
   ;
   ; This pseudo-MIR uses named metadata identifiers (e.g. !foo) to identify
   ; unnamed metadata (e.g. !0). To translate to MIR assign each unique named
   ; metadata identifier a monotonically increasing unnamed metadata identifier,
   ; then replace all references to each named metadata identifier with its
   ; corresponding unnamed metadata identifier.
   ;
   ; The identifiers are named as a dot (`.`) separated list of elements,
   ; ending with a tag corresponding to the type of metadata they identify.
   ;
   ; In MIR a `!DIExpr` is always printed inline at its use, even though it is
   ; internally uniqued and shared by all uses of the same expression. In this
   ; pseudo-MIR we break this convention and write the expressions out-of-line
   ; in some cases to emphasize where sharing occurs and to shorten the listing.

     lex_start:
       ; NOTE: These lifetimes for the PC/EXEC registers define the typical,
       ; default case of referring directly to the physical register. For cases
       ; like WQM where the physical EXEC and "logical" EXEC are not the same,
       ; this will be overriden by defining a bounded lifetime for
       ; !pc.fragment/!exec.fragment.
       DBG_DEF !pc.physical.lifetime, $PC
       DBG_DEF !exec.physical.lifetime, $EXEC
       DBG_DEF !bounded_divergent_lane_pc.lex.a.lifetime, $noreg
       a;
       %1 = EXEC;
       DBG_DEF !save_exec.lex_1.lifetime, u64 %1
       %2 = c1;
       DBG_KILL !bounded_divergent_lane_pc.lex.a.lifetime
     lex_1_start:
       DBG_LABEL !lex_1_start.label
       EXEC = %1 & %2;
     lex_1_then:
         DBG_DEF !bounded_divergent_lane_pc.lex_1_then.a.lifetime, $noreg
         b;
         %3 = EXEC;
         DBG_DEF !save_exec.lex_1_1.lifetime, u64 %3
         %4 = c2;
         DBG_KILL !bounded_divergent_lane_pc.lex_1_then.a.lifetime
     lex_1_1_start:
         DBG_LABEL !lex_1_1_start.label
         EXEC = %3 & %4;
     lex_1_1_then:
           DBG_DEF !bounded_divergent_lane_pc.lex_1_1_then.a.lifetime, $noreg
           c;
           DBG_KILL !bounded_divergent_lane_pc.lex_1_1_then.a.lifetime
         EXEC = ~EXEC & %3;
     lex_1_1_else:
           DBG_DEF !bounded_divergent_lane_pc.lex_1_1_else.a.lifetime, $noreg
           d;
           DBG_KILL !bounded_divergent_lane_pc.lex_1_1_else.a.lifetime
         EXEC = %3;
         DBG_KILL !save_exec.lex_1_1.lifetime
     lex_1_1_end:
         DBG_LABEL !lex_1_1_end.label
         DBG_DEF !bounded_divergent_lane_pc.lex_1_then.b.lifetime, $noreg
         e;
         DBG_KILL !bounded_divergent_lane_pc.lex_1_then.b.lifetime
       EXEC = ~EXEC & %1;
     lex_1_else:
         DBG_DEF !bounded_divergent_lane_pc.lex_1_else.a.lifetime, $noreg
         f;
         DBG_KILL !bounded_divergent_lane_pc.lex_1_else.a.lifetime
       EXEC = %1;
       DBG_KILL !save_exec.lex_1.lifetime
     lex_1_end:
       DBG_LABEL !lex_1_end.label
       DBG_DEF !bounded_divergent_lane_pc.lex.b.lifetime, $noreg
       g;
     lex_end:

   ;; Labels
   !lex_1_start.label = distinct !DExprCode()
   !lex_1_1_start.label = distinct !DExprCode()
   !lex_1_1_end.label = distinct !DExprCode()
   !lex_1_end.label = distinct !DExprCode()

   ;; Saved EXEC Mask Fragments
   ; These track the value of the EXEC mask saved on entry to each `IF/THEN/ELSE`
   ; region. The saved mask identifies the lanes to be updated when defining the
   ; computed divergent_lane_pc for a given lexical block (or, put another way,
   ; the negation of the saved mask identifies the lanes which are not updated).
   !save_exec.lex_1.fragment = distinct !DIFragment()
   !save_exec.lex_1.lifetime = distinct !DILifetime(
     object: !save_exec.lex_1.fragment,
     location: !DIExpr(DIOpReferrer(u64))
   )
   !save_exec.lex_1_1.fragment = distinct !DIFragment()
   !save_exec.lex_1_1.lifetime = distinct !DILifetime(
     object: !save_exec.lex_1_1.fragment,
     location: !DIExpr(DIOpReferrer(u64))
   )

   ;; Logical and Physical Register Fragments
   ; NOTE: We refer to the "logical" EXEC, `!exec.fragment`, in other expressions.
   ; This may be computed in cases where the physical EXEC was updated to
   ; implement e.g. whole-quad-mode. Referring to this fragment makes the uses
   ; transparently support this. The same approach is applied for the PC.
   !pc.fragment = distinct !DIFragment()
   !pc.default.lifetime = distinct !DILifetime(
     object: !pc.fragment,
     location: !DIExpr(DIOpArg(u64)),
     argObjects: {!pc.physical.fragment}
   )
   !pc.physical.fragment = distinct !DIFragment()
   !pc.physical.lifetime = distinct !DILifetime(
     object: !pc.physical.fragment,
     location: !DIExpr(DIOpReferrer(u64))
   )
   !exec.fragment = distinct !DIFragment()
   !exec.default.lifetime = distinct !DILifetime(
     object: !exec.fragment,
     location: !DIExpr(DIOpArg(u64)),
     argObjects: {!exec.physical.fragment}
   )
   !exec.physical.fragment = distinct !DIFragment()
   !exec.physical.lifetime = distinct !DILifetime(
     object: !exec.physical.fragment,
     location: !DIExpr(DIOpReferrer(u64))
   )

   ;; Bounded Divergent Lane PC
   ; This fragment has disjoint lifetimes which cover the entire PC range of the
   ; function. It contains the divergent_lane_pc for all lanes which are
   ; divergent, with unspecified values present in active lanes (as an artifact of
   ; the current implementation, the active lanes are assigned the same value as
   ; the divergent lanes which were active on entry to the current `IF/THEN/ELSE`
   ; region, but this is neither guaranteed nor required).
   !bounded_divergent_lane_pc.fragment = distinct !DIFragment()
   ; The argObjects to !bounded_divergent_lane_pc.expr are:
   ; {<64 x u64> lane_pc_vec}
   !bounded_divergent_lane_pc.expr = !DIExpr(DIOpArg(<64 x u64>))
   !bounded_divergent_lane_pc.lex.a.lifetime = distinct !DILifetime(
     object: !bounded_divergent_lane_pc.fragment,
     location: !bounded_divergent_lane_pc.expr,
     argObjects: {!divergent_lane_pc.lex.fragment}
   )
   !bounded_divergent_lane_pc.lex_1_then.a.lifetime = distinct !DILifetime(
     object: !bounded_divergent_lane_pc.fragment,
     location: !bounded_divergent_lane_pc.expr,
     argObjects: {!divergent_lane_pc.lex_1_then.fragment}
   )
   !bounded_divergent_lane_pc.lex_1_1_then.a.lifetime = distinct !DILifetime(
     object: !bounded_divergent_lane_pc.fragment,
     location: !bounded_divergent_lane_pc.expr,
     argObjects: {!divergent_lane_pc.lex_1_1_then.fragment}
   )
   !bounded_divergent_lane_pc.lex_1_1_else.a.lifetime = distinct !DILifetime(
     object: !bounded_divergent_lane_pc.fragment,
     location: !bounded_divergent_lane_pc.expr,
     argObjects: {!divergent_lane_pc.lex_1_1_else.fragment}
   )
   !bounded_divergent_lane_pc.lex_1_then.b.lifetime = distinct !DILifetime(
     object: !bounded_divergent_lane_pc.fragment,
     location: !bounded_divergent_lane_pc.expr,
     argObjects: {!divergent_lane_pc.lex_1_then.fragment}
   )
   !bounded_divergent_lane_pc.lex_1_else.a.lifetime = distinct !DILifetime(
     object: !bounded_divergent_lane_pc.fragment,
     location: !bounded_divergent_lane_pc.expr,
     argObjects: {!divergent_lane_pc.lex_1_else.fragment}
   )
   !bounded_divergent_lane_pc.lex.b.lifetime = distinct !DILifetime(
     object: !bounded_divergent_lane_pc.fragment,
     location: !bounded_divergent_lane_pc.expr,
     argObjects: {!divergent_lane_pc.lex.fragment}
   )

   ; TODO: Maybe add a property of DIFragment that asserts it should never have
   ; more than a single location description for any PC

   ; TODO: To easily translate Extend, Select, Read, etc.
   ; into DWARF, they will needs a type parameter. Should we add a type to just the
   ; operations which correspond to a DWARF operation that needs the type/size? Or
   ; should we just add types to all operations?

   ;; Computed Divergent Lane PC Fragments
   !divergent_lane_pc.lex.fragment = distinct !DIFragment()
   !divergent_lane_pc.lex.lifetime = distinct !DILifetime(
     object: !divergent_lane_pc_outer.fragment,
     location: !DIExpr(DIOpConstant(u64 undef), DIOpExtend(64))
   )
   ; The argObjects to `!select_lanes.expr` are:
   ; {<64 x u64> starting_lane_pc_vec, u64 pc_value, u64 mask}
   !select_lanes.expr = !DIExpr(
     DIOpArg(0, <64 x u64>),
     DIOpArg(1, u64), DIOpExtend(64, u64),
     DIOpArg(2, u64),
     DIOpSelect(64, u64)
   )
   ; TODO: We have the issue of: how do we ensure we have a value when we need
   ; it for DWARF, for example DIOpSelect will need to ensure the top element of
   ; the stack is a value when evaluating the final DWARF, but this violates the
   ; "context insensitive" property we want for the operations.
   ; We can work around this by emitting "unoptimized" DWARF where e.g. every
   ; implicit location description in the LLVM representation actually maps to an
   ; implicit location description being pushed on the DWARF stack (e.g. we lower
   ; `... DIOpConstant(u64 42) DIOpSelect()` to `... DW_OP_uconst 42,
   ; DW_OP_stack_value, DW_OP_deref, DW_OP_select_bit_piece` instead of just `...
   ; DW_OP_uconst 42, DW_OP_select_bit_piece`)
   !divergent_lane_pc.lex_1_then.fragment = distinct !DIFragment()
   !divergent_lane_pc.lex_1_then.lifetime = distinct !DILifetime(
     object: !divergent_lane_pc.lex_1_then.fragment,
     location: !select_lanes.expr,
     argObjects: {
       !divergent_lane_pc.lex.fragment,
       !lex_1_start.label,
       !save_exec.lex_1.fragment
     }
   )
   !divergent_lane_pc.lex_1_1_then.fragment = distinct !DIFragment()
   !divergent_lane_pc.lex_1_1_then.lifetime = distinct !DILifetime(
     object: !divergent_lane_pc.lex_1_1_then.fragment,
     location: !select_lanes.expr,
     argObjects: {
       !divergent_lane_pc.lex.fragment,
       !lex_1_1_start.label,
       !save_exec.lex_1_1.fragment
     }
   )
   !divergent_lane_pc.lex_1_1_else.fragment = distinct !DIFragment()
   !divergent_lane_pc.lex_1_1_else.lifetime = distinct !DILifetime(
     object: !divergent_lane_pc.lex_1_1_else.fragment,
     location: !select_lanes.expr,
     argObjects: {
       !divergent_lane_pc.lex.fragment,
       !lex_1_1_end.label,
       !save_exec.lex_1_1.fragment
     }
   )
   !divergent_lane_pc.lex_1_else.fragment = distinct !DIFragment()
   !divergent_lane_pc.lex_1_else.lifetime = distinct !DILifetime(
     object: !divergent_lane_pc.lex_1_else.fragment,
     location: !select_lanes.expr,
     argObjects: {
       !divergent_lane_pc.lex.fragment,
       !lex_1_end.label,
       !save_exec.lex_1.fragment
     }
   )

   ;; Active Lane PC
   !active_lane_pc.fragment = distinct !DIFragment()
   !active_lane_pc.lifetime = distinct !DILifetime(
     object: !active_lane_pc.fragment,
     location: !select_lanes.expr,
     argObjects: {
       !bounded_divergent_lane_pc.fragment,
       !pc.fragment,
       !exec.fragment
     }
   )

   ;; Subprogram
   !subprogram = !DISubprogram(...,
     activeLanePC: !active_lane_pc.fragment,
     retainedNodes: !{
       !pc.default.lifetime,
       !exec.default.lifetime,
       !divergent_lane_pc.lex_1_then.lifetime,
       !divergent_lane_pc.lex_1_1_then.lifetime,
       !divergent_lane_pc.lex_1_1_else.lifetime,
       !divergent_lane_pc.lex_1_else.lifetime,
       !active_lane_pc.lifetime,
       !lex_1_start.label,
       !lex_1_1_start.label,
       !lex_1_1_end.label,
       !lex_1_end.label
     }
   )

Fragments ``!save_exec.lex_1.fragment`` and ``!save_exec.lex_1_1.fragment`` are
created for the execution masks saved on entry to a region. Using the
``DBG_DEF`` pseudo instruction, location list entries will be created that
describe where the artificial variables are allocated at any given program
location. The compiler may allocate them to registers or spill them to memory.

The fragments for each region use the values of the saved execution mask
artificial variables to only update the lanes that are active on entry to the
region. All other lanes retain the value of the enclosing region where they were
last active. If they were not active on entry to the subprogram, then will have
the undefined location description.

Other structured control flow regions can be handled similarly. For example,
loops would set the divergent program location for the region at the end of the
loop. Any lanes active will be in the loop, and any lanes not active must have
exited the loop.

An ``IF/THEN/ELSEIF/ELSEIF/...`` region can be treated as a nest of
``IF/THEN/ELSE`` regions.

Other Ideas
===========

Translating To DWARF
--------------------

.. TODO:::

   Define algorithm for computing DWARF location descriptions and loclists.

   -  Define rule for implicit pointers (``DIOpAddrof`` operation applied to a
      ``DIOpReferrer`` operation):

      -  Look for a compatible, existing program object.
      -  If not, generate an artificial one.
      -  This could be bubbled up to DWARF itself, to allow implicits to hold
         arbitrary location descriptions, eliminating the need for the
         artificial variable, and make translation simpler.

   -  Define rule for ``DIFragment``:

      -  If referenced by multiple ``argObjects``, then use a
         ``DW_TAG_DWARF_procedure``.
      -  If only referenced by a ``DIVariable`` or ``DIComposite`` field, then
         use ``expr`` or ``loclist`` form that specifies the location
         description expression directly.

   -  Define rule for computed lifetime:

      -  If referenced ``DIObject`` has no bounded lifetime segments, then use
         ``expr`` form.
      -  If referenced ``DIObject`` has bounded lifetime segments, then use
         ``loclist`` form.

Translating To PDB (CodeView)
-----------------------------

.. TODO::

   Define.

Comparison With GCC
-------------------

.. TODO::

   Understand how this compares to what GCC is doing?

Example Ideas
-------------

Spilling
~~~~~~~~

.. TODO::

   SSA -> stack slot

.. code:: llvm
   :number-lines:

   %x = i32 ...
   call void @llvm.dbg.def(metadata !1, metadata i32 %x)
   ...
   call void @llvm.dbg.kill(metadata !1)

   !0 = !DILocalVariable("x")
   !1 = distinct !DILifetime(object: !0, location: !DIExpr(DIOpReferrer(i32)))

spill %x:

.. code:: llvm
   :number-lines:

   %x.addr = alloca i32, addrspace(5)
   store i32* %x.addr, ...
   call void @llvm.dbg.def(metadata !1, metadata i32 *%x)
   ...
   call void @llvm.dbg.kill(metadata !1)

   !0 = !DILocalVariable("x")
   !1 = distinct !DILifetime(object: !0, location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)))

..

.. TODO::

   stack slot -> register

..

.. TODO::

   register -> stack slot

Simultaneous Lifetimes In Multiple Places
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO::

   Define.

File Scope Globals
~~~~~~~~~~~~~~~~~~

.. TODO::

   Define.

LDS Variables
~~~~~~~~~~~~~

.. TODO::

   LDS variables, one variable but multiple kernels with distinct lifetimes, is
   that possible in LLVM?

   Could allow the ``llvm.dbg.def`` intrinsic to refer to a global and use that
   to define live ranges which live in functions and refer to storage outside of
   the function.

   I would expect that LDS variables would have no ``!dbg.default`` and instead
   have ``llvm.dbg.def`` in each function that can access it. The bounded
   lifetime segment would have an expression that evaluates to the location of
   the LDS variable in the specific subprogram. For a kernel it would likely be
   an absolute address in the LDS address space. Each kernel may have a
   different address. In functions that can be called from multiple kernels it
   may be an expression that uses the LDS indirection variables to determine the
   actual LDS address.

Make Sure The Non-SSA MIR Form Works With def/kill Scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO::

   Make sure the non-SSA MIR form works with def/kill scheme, and additionally
   confirm why we do not seem to need the work upstream that is trying to move
   to referring to an instruction rather than a register? See `[llvm-dev] [RFC]
   DebugInfo: A different way of specifying variable locations post-isel
   <https://lists.llvm.org/pipermail/llvm-dev/2020-February/139440.html>`__.

Integer Fragment IDs
--------------------

.. TODO::

   This was just a quick jotting-down of one idea for eliminating the need for a
   distinct ``DIFragment`` to represent the identity of fragments.

.. _local-variable-broken-into-two-scalars-1:

Local Variable Broken Into Two Scalars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: llvm
   :number-lines:

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

Further Decomposition Of An Already SRoA’d Local Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: llvm
   :number-lines:

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

Multiple Live Ranges For A Fragment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: llvm
   :number-lines:

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

References
==========

1.  `[LLVMdev] [RFC] Separating Metadata from the Value hierarchy (David
    Blaikie)
    <https://lists.llvm.org/pipermail/llvm-dev/2014-November/078656.html>`__

2.  `[LLVMdev] [RFC] Separating Metadata from the Value hierarchy
    <https://lists.llvm.org/pipermail/llvm-dev/2014-November/078682.html>`__

3.  `[llvm-dev] Proposal for multi location debug info support in LLVM IR <https://lists.llvm.org/pipermail/llvm-dev/2015-December/093535.html>`__

4.  `[llvm-dev] Proposal for multi location debug info support in LLVM IR <https://lists.llvm.org/pipermail/llvm-dev/2016-January/093627.html>`__

5.  `Multi Location Debug Info support for LLVM <https://gist.github.com/Keno/480b8057df1b7c63c321>`__

6.  `D81852 [DebugInfo] Update MachineInstr interface to better support variadic DBG_VALUE instructions <https://reviews.llvm.org/D81852>`__

7.  `D70601 Disallow DIExpressions with shift operators from being fragmented <https://reviews.llvm.org/D70601>`__

8.  `D57962 [DebugInfo] PR40628: Don’t salvage load operations <https://reviews.llvm.org/D57962>`__

9.  `Bug 40628 - [DebugInfo@O2] Salvaged memory loads can observe subsequent memory writes <https://bugs.llvm.org/show_bug.cgi?id=40628>`__

10. :doc:`LangRef`

    1. :ref:`wellformed`
    2. :ref:`typesystem`
    3. :ref:`globalvars`
    4. :ref:`DICompositeType`
    5. :ref:`DILocalVariable`
    6. :ref:`DIGlobalVariable`
    7. :ref:`DICompileUnit`
    8. :ref:`DISubprogram`
    9. :ref:`DILabel`

11. :doc:`AMDGPUDwarfExtensionsForHeterogeneousDebugging`

    1. :ref:`amdgpu-dwarf-expressions`
    2. :ref:`amdgpu-dwarf-location-list-expressions`
    3. :ref:`amdgpu-dwarf-location-description`
    4. :ref:`amdgpu-dwarf-expression-evaluation-context`

12. :doc:`AMDGPUUsage`

    1. :ref:`amdgpu-dwarf-dw-at-llvm-lane-pc`
