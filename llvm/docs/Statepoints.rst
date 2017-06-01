=====================================
Garbage Collection Safepoints in LLVM
=====================================

.. contents::
   :local:
   :depth: 2

Status
=======

This document describes a set of experimental extensions to LLVM. Use
with caution.  Because the intrinsics have experimental status,
compatibility across LLVM releases is not guaranteed.

LLVM currently supports an alternate mechanism for conservative
garbage collection support using the ``gcroot`` intrinsic.  The mechanism
described here shares little in common with the alternate ``gcroot``
implementation and it is hoped that this mechanism will eventually
replace the gc_root mechanism.

Overview
========

To collect dead objects, garbage collectors must be able to identify
any references to objects contained within executing code, and,
depending on the collector, potentially update them.  The collector
does not need this information at all points in code - that would make
the problem much harder - but only at well-defined points in the
execution known as 'safepoints' For most collectors, it is sufficient
to track at least one copy of each unique pointer value.  However, for
a collector which wishes to relocate objects directly reachable from
running code, a higher standard is required.

One additional challenge is that the compiler may compute intermediate
results ("derived pointers") which point outside of the allocation or
even into the middle of another allocation.  The eventual use of this
intermediate value must yield an address within the bounds of the
allocation, but such "exterior derived pointers" may be visible to the
collector.  Given this, a garbage collector can not safely rely on the
runtime value of an address to indicate the object it is associated
with.  If the garbage collector wishes to move any object, the
compiler must provide a mapping, for each pointer, to an indication of
its allocation.

To simplify the interaction between a collector and the compiled code,
most garbage collectors are organized in terms of three abstractions:
load barriers, store barriers, and safepoints.

#. A load barrier is a bit of code executed immediately after the
   machine load instruction, but before any use of the value loaded.
   Depending on the collector, such a barrier may be needed for all
   loads, merely loads of a particular type (in the original source
   language), or none at all.

#. Analogously, a store barrier is a code fragment that runs
   immediately before the machine store instruction, but after the
   computation of the value stored.  The most common use of a store
   barrier is to update a 'card table' in a generational garbage
   collector.

#. A safepoint is a location at which pointers visible to the compiled
   code (i.e. currently in registers or on the stack) are allowed to
   change.  After the safepoint completes, the actual pointer value
   may differ, but the 'object' (as seen by the source language)
   pointed to will not.

  Note that the term 'safepoint' is somewhat overloaded.  It refers to
  both the location at which the machine state is parsable and the
  coordination protocol involved in bring application threads to a
  point at which the collector can safely use that information.  The
  term "statepoint" as used in this document refers exclusively to the
  former.

This document focuses on the last item - compiler support for
safepoints in generated code.  We will assume that an outside
mechanism has decided where to place safepoints.  From our
perspective, all safepoints will be function calls.  To support
relocation of objects directly reachable from values in compiled code,
the collector must be able to:

#. identify every copy of a pointer (including copies introduced by
   the compiler itself) at the safepoint,
#. identify which object each pointer relates to, and
#. potentially update each of those copies.

This document describes the mechanism by which an LLVM based compiler
can provide this information to a language runtime/collector, and
ensure that all pointers can be read and updated if desired.  The
heart of the approach is to construct (or rewrite) the IR in a manner
where the possible updates performed by the garbage collector are
explicitly visible in the IR.  Doing so requires that we:

#. create a new SSA value for each potentially relocated pointer, and
   ensure that no uses of the original (non relocated) value is
   reachable after the safepoint,
#. specify the relocation in a way which is opaque to the compiler to
   ensure that the optimizer can not introduce new uses of an
   unrelocated value after a statepoint. This prevents the optimizer
   from performing unsound optimizations.
#. recording a mapping of live pointers (and the allocation they're
   associated with) for each statepoint.

At the most abstract level, inserting a safepoint can be thought of as
replacing a call instruction with a call to a multiple return value
function which both calls the original target of the call, returns
it's result, and returns updated values for any live pointers to
garbage collected objects.

  Note that the task of identifying all live pointers to garbage
  collected values, transforming the IR to expose a pointer giving the
  base object for every such live pointer, and inserting all the
  intrinsics correctly is explicitly out of scope for this document.
  The recommended approach is to use the :ref:`utility passes 
  <statepoint-utilities>` described below. 

This abstract function call is concretely represented by a sequence of
intrinsic calls known collectively as a "statepoint relocation sequence".

Let's consider a simple call in LLVM IR:

.. code-block:: llvm

  define i8 addrspace(1)* @test1(i8 addrspace(1)* %obj) 
         gc "statepoint-example" {
    call void ()* @foo()
    ret i8 addrspace(1)* %obj
  }

Depending on our language we may need to allow a safepoint during the execution 
of ``foo``. If so, we need to let the collector update local values in the 
current frame.  If we don't, we'll be accessing a potential invalid reference 
once we eventually return from the call.

In this example, we need to relocate the SSA value ``%obj``.  Since we can't 
actually change the value in the SSA value ``%obj``, we need to introduce a new 
SSA value ``%obj.relocated`` which represents the potentially changed value of
``%obj`` after the safepoint and update any following uses appropriately.  The 
resulting relocation sequence is:

.. code-block:: text

  define i8 addrspace(1)* @test1(i8 addrspace(1)* %obj) 
         gc "statepoint-example" {
    %0 = call token (i64, i32, void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0, i8 addrspace(1)* %obj)
    %obj.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %0, i32 7, i32 7)
    ret i8 addrspace(1)* %obj.relocated
  }

Ideally, this sequence would have been represented as a M argument, N
return value function (where M is the number of values being
relocated + the original call arguments and N is the original return
value + each relocated value), but LLVM does not easily support such a
representation.

Instead, the statepoint intrinsic marks the actual site of the
safepoint or statepoint.  The statepoint returns a token value (which
exists only at compile time).  To get back the original return value
of the call, we use the ``gc.result`` intrinsic.  To get the relocation
of each pointer in turn, we use the ``gc.relocate`` intrinsic with the
appropriate index.  Note that both the ``gc.relocate`` and ``gc.result`` are
tied to the statepoint.  The combination forms a "statepoint relocation 
sequence" and represents the entirety of a parseable call or 'statepoint'.

When lowered, this example would generate the following x86 assembly:

.. code-block:: gas
  
	  .globl	test1
	  .align	16, 0x90
	  pushq	%rax
	  callq	foo
  .Ltmp1:
	  movq	(%rsp), %rax  # This load is redundant (oops!)
	  popq	%rdx
	  retq

Each of the potentially relocated values has been spilled to the
stack, and a record of that location has been recorded to the
:ref:`Stack Map section <stackmap-section>`.  If the garbage collector
needs to update any of these pointers during the call, it knows
exactly what to change.

The relevant parts of the StackMap section for our example are:

.. code-block:: gas
  
  # This describes the call site
  # Stack Maps: callsite 2882400000
	  .quad	2882400000
	  .long	.Ltmp1-test1
	  .short	0
  # .. 8 entries skipped ..
  # This entry describes the spill slot which is directly addressable
  # off RSP with offset 0.  Given the value was spilled with a pushq, 
  # that makes sense.
  # Stack Maps:   Loc 8: Direct RSP     [encoding: .byte 2, .byte 8, .short 7, .int 0]
	  .byte	2
	  .byte	8
	  .short	7
	  .long	0

This example was taken from the tests for the :ref:`RewriteStatepointsForGC` utility pass.  As such, it's full StackMap can be easily examined with the following command.

.. code-block:: bash

  opt -rewrite-statepoints-for-gc test/Transforms/RewriteStatepointsForGC/basics.ll -S | llc -debug-only=stackmaps

Base & Derived Pointers
^^^^^^^^^^^^^^^^^^^^^^^

A "base pointer" is one which points to the starting address of an allocation
(object).  A "derived pointer" is one which is offset from a base pointer by
some amount.  When relocating objects, a garbage collector needs to be able 
to relocate each derived pointer associated with an allocation to the same 
offset from the new address.

"Interior derived pointers" remain within the bounds of the allocation 
they're associated with.  As a result, the base object can be found at 
runtime provided the bounds of allocations are known to the runtime system.

"Exterior derived pointers" are outside the bounds of the associated object;
they may even fall within *another* allocations address range.  As a result,
there is no way for a garbage collector to determine which allocation they 
are associated with at runtime and compiler support is needed.

The ``gc.relocate`` intrinsic supports an explicit operand for describing the
allocation associated with a derived pointer.  This operand is frequently 
referred to as the base operand, but does not strictly speaking have to be
a base pointer, but it does need to lie within the bounds of the associated
allocation.  Some collectors may require that the operand be an actual base
pointer rather than merely an internal derived pointer. Note that during 
lowering both the base and derived pointer operands are required to be live 
over the associated call safepoint even if the base is otherwise unused 
afterwards.

If we extend our previous example to include a pointless derived pointer, 
we get:

.. code-block:: text

  define i8 addrspace(1)* @test1(i8 addrspace(1)* %obj) 
         gc "statepoint-example" {
    %gep = getelementptr i8, i8 addrspace(1)* %obj, i64 20000
    %token = call token (i64, i32, void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0, i8 addrspace(1)* %obj, i8 addrspace(1)* %gep)
    %obj.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %token, i32 7, i32 7)
    %gep.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %token, i32 7, i32 8)
    %p = getelementptr i8, i8 addrspace(1)* %gep, i64 -20000
    ret i8 addrspace(1)* %p
  }

Note that in this example %p and %obj.relocate are the same address and we
could replace one with the other, potentially removing the derived pointer
from the live set at the safepoint entirely.

.. _gc_transition_args:

GC Transitions
^^^^^^^^^^^^^^^^^^

As a practical consideration, many garbage-collected systems allow code that is
collector-aware ("managed code") to call code that is not collector-aware
("unmanaged code"). It is common that such calls must also be safepoints, since
it is desirable to allow the collector to run during the execution of
unmanaged code. Furthermore, it is common that coordinating the transition from
managed to unmanaged code requires extra code generation at the call site to
inform the collector of the transition. In order to support these needs, a
statepoint may be marked as a GC transition, and data that is necessary to
perform the transition (if any) may be provided as additional arguments to the
statepoint.

  Note that although in many cases statepoints may be inferred to be GC
  transitions based on the function symbols involved (e.g. a call from a
  function with GC strategy "foo" to a function with GC strategy "bar"),
  indirect calls that are also GC transitions must also be supported. This
  requirement is the driving force behind the decision to require that GC
  transitions are explicitly marked.

Let's revisit the sample given above, this time treating the call to ``@foo``
as a GC transition. Depending on our target, the transition code may need to
access some extra state in order to inform the collector of the transition.
Let's assume a hypothetical GC--somewhat unimaginatively named "hypothetical-gc"
--that requires that a TLS variable must be written to before and after a call
to unmanaged code. The resulting relocation sequence is:

.. code-block:: text

  @flag = thread_local global i32 0, align 4

  define i8 addrspace(1)* @test1(i8 addrspace(1) *%obj)
         gc "hypothetical-gc" {

    %0 = call token (i64, i32, void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 1, i32* @Flag, i32 0, i8 addrspace(1)* %obj)
    %obj.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %0, i32 7, i32 7)
    ret i8 addrspace(1)* %obj.relocated
  }

During lowering, this will result in a instruction selection DAG that looks
something like:

::

  CALLSEQ_START
  ...
  GC_TRANSITION_START (lowered i32 *@Flag), SRCVALUE i32* Flag
  STATEPOINT
  GC_TRANSITION_END (lowered i32 *@Flag), SRCVALUE i32 *Flag
  ...
  CALLSEQ_END

In order to generate the necessary transition code, the backend for each target
supported by "hypothetical-gc" must be modified to lower ``GC_TRANSITION_START``
and ``GC_TRANSITION_END`` nodes appropriately when the "hypothetical-gc"
strategy is in use for a particular function. Assuming that such lowering has
been added for X86, the generated assembly would be:

.. code-block:: gas

	  .globl	test1
	  .align	16, 0x90
	  pushq	%rax
	  movl $1, %fs:Flag@TPOFF
	  callq	foo
	  movl $0, %fs:Flag@TPOFF
  .Ltmp1:
	  movq	(%rsp), %rax  # This load is redundant (oops!)
	  popq	%rdx
	  retq

Note that the design as presented above is not fully implemented: in particular,
strategy-specific lowering is not present, and all GC transitions are emitted as
as single no-op before and after the call instruction. These no-ops are often
removed by the backend during dead machine instruction elimination.


Intrinsics
===========

'llvm.experimental.gc.statepoint' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare token
        @llvm.experimental.gc.statepoint(i64 <id>, i32 <num patch bytes>,
                       func_type <target>, 
                       i64 <#call args>, i64 <flags>,
                       ... (call parameters),
                       i64 <# transition args>, ... (transition parameters),
                       i64 <# deopt args>, ... (deopt parameters),
                       ... (gc parameters))

Overview:
"""""""""

The statepoint intrinsic represents a call which is parse-able by the
runtime.

Operands:
"""""""""

The 'id' operand is a constant integer that is reported as the ID
field in the generated stackmap.  LLVM does not interpret this
parameter in any way and its meaning is up to the statepoint user to
decide.  Note that LLVM is free to duplicate code containing
statepoint calls, and this may transform IR that had a unique 'id' per
lexical call to statepoint to IR that does not.

If 'num patch bytes' is non-zero then the call instruction
corresponding to the statepoint is not emitted and LLVM emits 'num
patch bytes' bytes of nops in its place.  LLVM will emit code to
prepare the function arguments and retrieve the function return value
in accordance to the calling convention; the former before the nop
sequence and the latter after the nop sequence.  It is expected that
the user will patch over the 'num patch bytes' bytes of nops with a
calling sequence specific to their runtime before executing the
generated machine code.  There are no guarantees with respect to the
alignment of the nop sequence.  Unlike :doc:`StackMaps` statepoints do
not have a concept of shadow bytes.  Note that semantically the
statepoint still represents a call or invoke to 'target', and the nop
sequence after patching is expected to represent an operation
equivalent to a call or invoke to 'target'.

The 'target' operand is the function actually being called.  The
target can be specified as either a symbolic LLVM function, or as an
arbitrary Value of appropriate function type.  Note that the function
type must match the signature of the callee and the types of the 'call
parameters' arguments.

The '#call args' operand is the number of arguments to the actual
call.  It must exactly match the number of arguments passed in the
'call parameters' variable length section.

The 'flags' operand is used to specify extra information about the
statepoint. This is currently only used to mark certain statepoints
as GC transitions. This operand is a 64-bit integer with the following
layout, where bit 0 is the least significant bit:

  +-------+---------------------------------------------------+
  | Bit # | Usage                                             |
  +=======+===================================================+
  |     0 | Set if the statepoint is a GC transition, cleared |
  |       | otherwise.                                        |
  +-------+---------------------------------------------------+
  |  1-63 | Reserved for future use; must be cleared.         |
  +-------+---------------------------------------------------+

The 'call parameters' arguments are simply the arguments which need to
be passed to the call target.  They will be lowered according to the
specified calling convention and otherwise handled like a normal call
instruction.  The number of arguments must exactly match what is
specified in '# call args'.  The types must match the signature of
'target'.

The 'transition parameters' arguments contain an arbitrary list of
Values which need to be passed to GC transition code. They will be
lowered and passed as operands to the appropriate GC_TRANSITION nodes
in the selection DAG. It is assumed that these arguments must be
available before and after (but not necessarily during) the execution
of the callee. The '# transition args' field indicates how many operands
are to be interpreted as 'transition parameters'.

The 'deopt parameters' arguments contain an arbitrary list of Values
which is meaningful to the runtime.  The runtime may read any of these
values, but is assumed not to modify them.  If the garbage collector
might need to modify one of these values, it must also be listed in
the 'gc pointer' argument list.  The '# deopt args' field indicates
how many operands are to be interpreted as 'deopt parameters'.

The 'gc parameters' arguments contain every pointer to a garbage
collector object which potentially needs to be updated by the garbage
collector.  Note that the argument list must explicitly contain a base
pointer for every derived pointer listed.  The order of arguments is
unimportant.  Unlike the other variable length parameter sets, this
list is not length prefixed.

Semantics:
""""""""""

A statepoint is assumed to read and write all memory.  As a result,
memory operations can not be reordered past a statepoint.  It is
illegal to mark a statepoint as being either 'readonly' or 'readnone'.

Note that legal IR can not perform any memory operation on a 'gc
pointer' argument of the statepoint in a location statically reachable
from the statepoint.  Instead, the explicitly relocated value (from a
``gc.relocate``) must be used.

'llvm.experimental.gc.result' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare type*
        @llvm.experimental.gc.result(token %statepoint_token)

Overview:
"""""""""

``gc.result`` extracts the result of the original call instruction
which was replaced by the ``gc.statepoint``.  The ``gc.result``
intrinsic is actually a family of three intrinsics due to an
implementation limitation.  Other than the type of the return value,
the semantics are the same.

Operands:
"""""""""

The first and only argument is the ``gc.statepoint`` which starts
the safepoint sequence of which this ``gc.result`` is a part.
Despite the typing of this as a generic token, *only* the value defined 
by a ``gc.statepoint`` is legal here.

Semantics:
""""""""""

The ``gc.result`` represents the return value of the call target of
the ``statepoint``.  The type of the ``gc.result`` must exactly match
the type of the target.  If the call target returns void, there will
be no ``gc.result``.

A ``gc.result`` is modeled as a 'readnone' pure function.  It has no
side effects since it is just a projection of the return value of the
previous call represented by the ``gc.statepoint``.

'llvm.experimental.gc.relocate' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare <pointer type>
        @llvm.experimental.gc.relocate(token %statepoint_token, 
                                       i32 %base_offset, 
                                       i32 %pointer_offset)

Overview:
"""""""""

A ``gc.relocate`` returns the potentially relocated value of a pointer
at the safepoint.

Operands:
"""""""""

The first argument is the ``gc.statepoint`` which starts the
safepoint sequence of which this ``gc.relocation`` is a part.
Despite the typing of this as a generic token, *only* the value defined 
by a ``gc.statepoint`` is legal here.

The second argument is an index into the statepoints list of arguments
which specifies the allocation for the pointer being relocated.
This index must land within the 'gc parameter' section of the
statepoint's argument list.  The associated value must be within the
object with which the pointer being relocated is associated. The optimizer
is free to change *which* interior derived pointer is reported, provided that
it does not replace an actual base pointer with another interior derived 
pointer.  Collectors are allowed to rely on the base pointer operand 
remaining an actual base pointer if so constructed.

The third argument is an index into the statepoint's list of arguments
which specify the (potentially) derived pointer being relocated.  It
is legal for this index to be the same as the second argument
if-and-only-if a base pointer is being relocated. This index must land
within the 'gc parameter' section of the statepoint's argument list.

Semantics:
""""""""""

The return value of ``gc.relocate`` is the potentially relocated value
of the pointer specified by it's arguments.  It is unspecified how the
value of the returned pointer relates to the argument to the
``gc.statepoint`` other than that a) it points to the same source
language object with the same offset, and b) the 'based-on'
relationship of the newly relocated pointers is a projection of the
unrelocated pointers.  In particular, the integer value of the pointer
returned is unspecified.

A ``gc.relocate`` is modeled as a ``readnone`` pure function.  It has no
side effects since it is just a way to extract information about work
done during the actual call modeled by the ``gc.statepoint``.

.. _statepoint-stackmap-format:

Stack Map Format
================

Locations for each pointer value which may need read and/or updated by
the runtime or collector are provided via the :ref:`Stack Map format
<stackmap-format>` specified in the PatchPoint documentation.

Each statepoint generates the following Locations:

* Constant which describes the calling convention of the call target. This
  constant is a valid :ref:`calling convention identifier <callingconv>` for
  the version of LLVM used to generate the stackmap. No additional compatibility
  guarantees are made for this constant over what LLVM provides elsewhere w.r.t.
  these identifiers.
* Constant which describes the flags passed to the statepoint intrinsic
* Constant which describes number of following deopt *Locations* (not
  operands)
* Variable number of Locations, one for each deopt parameter listed in
  the IR statepoint (same number as described by previous Constant).  At 
  the moment, only deopt parameters with a bitwidth of 64 bits or less 
  are supported.  Values of a type larger than 64 bits can be specified 
  and reported only if a) the value is constant at the call site, and b) 
  the constant can be represented with less than 64 bits (assuming zero 
  extension to the original bitwidth).
* Variable number of relocation records, each of which consists of 
  exactly two Locations.  Relocation records are described in detail
  below.

Each relocation record provides sufficient information for a collector to 
relocate one or more derived pointers.  Each record consists of a pair of 
Locations.  The second element in the record represents the pointer (or 
pointers) which need updated.  The first element in the record provides a 
pointer to the base of the object with which the pointer(s) being relocated is
associated.  This information is required for handling generalized derived 
pointers since a pointer may be outside the bounds of the original allocation,
but still needs to be relocated with the allocation.  Additionally:

* It is guaranteed that the base pointer must also appear explicitly as a 
  relocation pair if used after the statepoint. 
* There may be fewer relocation records then gc parameters in the IR
  statepoint. Each *unique* pair will occur at least once; duplicates
  are possible.  
* The Locations within each record may either be of pointer size or a 
  multiple of pointer size.  In the later case, the record must be 
  interpreted as describing a sequence of pointers and their corresponding 
  base pointers. If the Location is of size N x sizeof(pointer), then
  there will be N records of one pointer each contained within the Location.
  Both Locations in a pair can be assumed to be of the same size.

Note that the Locations used in each section may describe the same
physical location.  e.g. A stack slot may appear as a deopt location,
a gc base pointer, and a gc derived pointer.

The LiveOut section of the StkMapRecord will be empty for a statepoint
record.

Safepoint Semantics & Verification
==================================

The fundamental correctness property for the compiled code's
correctness w.r.t. the garbage collector is a dynamic one.  It must be
the case that there is no dynamic trace such that a operation
involving a potentially relocated pointer is observably-after a
safepoint which could relocate it.  'observably-after' is this usage
means that an outside observer could observe this sequence of events
in a way which precludes the operation being performed before the
safepoint.

To understand why this 'observable-after' property is required,
consider a null comparison performed on the original copy of a
relocated pointer.  Assuming that control flow follows the safepoint,
there is no way to observe externally whether the null comparison is
performed before or after the safepoint.  (Remember, the original
Value is unmodified by the safepoint.)  The compiler is free to make
either scheduling choice.

The actual correctness property implemented is slightly stronger than
this.  We require that there be no *static path* on which a
potentially relocated pointer is 'observably-after' it may have been
relocated.  This is slightly stronger than is strictly necessary (and
thus may disallow some otherwise valid programs), but greatly
simplifies reasoning about correctness of the compiled code.

By construction, this property will be upheld by the optimizer if
correctly established in the source IR.  This is a key invariant of
the design.

The existing IR Verifier pass has been extended to check most of the
local restrictions on the intrinsics mentioned in their respective
documentation.  The current implementation in LLVM does not check the
key relocation invariant, but this is ongoing work on developing such
a verifier.  Please ask on llvm-dev if you're interested in
experimenting with the current version.

.. _statepoint-utilities:

Utility Passes for Safepoint Insertion
======================================

.. _RewriteStatepointsForGC:

RewriteStatepointsForGC
^^^^^^^^^^^^^^^^^^^^^^^^

The pass RewriteStatepointsForGC transforms a functions IR by replacing a 
``gc.statepoint`` (with an optional ``gc.result``) with a full relocation 
sequence, including all required ``gc.relocates``.  To function, the pass 
requires that the GC strategy specified for the function be able to reliably 
distinguish between GC references and non-GC references in IR it is given.

As an example, given this code:

.. code-block:: text

  define i8 addrspace(1)* @test1(i8 addrspace(1)* %obj) 
         gc "statepoint-example" {
    call token (i64, i32, void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
    ret i8 addrspace(1)* %obj
  }

The pass would produce this IR:

.. code-block:: text

  define i8 addrspace(1)* @test1(i8 addrspace(1)* %obj) 
         gc "statepoint-example" {
    %0 = call token (i64, i32, void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i8 addrspace(1)* %obj)
    %obj.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %0, i32 12, i32 12)
    ret i8 addrspace(1)* %obj.relocated
  }

In the above examples, the addrspace(1) marker on the pointers is the mechanism
that the ``statepoint-example`` GC strategy uses to distinguish references from
non references.  Address space 1 is not globally reserved for this purpose.

This pass can be used an utility function by a language frontend that doesn't 
want to manually reason about liveness, base pointers, or relocation when 
constructing IR.  As currently implemented, RewriteStatepointsForGC must be 
run after SSA construction (i.e. mem2ref).

RewriteStatepointsForGC will ensure that appropriate base pointers are listed
for every relocation created.  It will do so by duplicating code as needed to
propagate the base pointer associated with each pointer being relocated to
the appropriate safepoints.  The implementation assumes that the following 
IR constructs produce base pointers: loads from the heap, addresses of global 
variables, function arguments, function return values. Constant pointers (such
as null) are also assumed to be base pointers.  In practice, this constraint
can be relaxed to producing interior derived pointers provided the target 
collector can find the associated allocation from an arbitrary interior 
derived pointer.

In practice, RewriteStatepointsForGC can be run much later in the pass 
pipeline, after most optimization is already done.  This helps to improve 
the quality of the generated code when compiled with garbage collection support.
In the long run, this is the intended usage model.  At this time, a few details
have yet to be worked out about the semantic model required to guarantee this 
is always correct.  As such, please use with caution and report bugs.

.. _PlaceSafepoints:

PlaceSafepoints
^^^^^^^^^^^^^^^^

The pass PlaceSafepoints transforms a function's IR by replacing any call or 
invoke instructions with appropriate ``gc.statepoint`` and ``gc.result`` pairs,
and inserting safepoint polls sufficient to ensure running code checks for a 
safepoint request on a timely manner.  This pass is expected to be run before 
RewriteStatepointsForGC and thus does not produce full relocation sequences.  

As an example, given input IR of the following:

.. code-block:: llvm

  define void @test() gc "statepoint-example" {
    call void @foo()
    ret void
  }

  declare void @do_safepoint()
  define void @gc.safepoint_poll() {
    call void @do_safepoint()
    ret void
  }


This pass would produce the following IR:

.. code-block:: text

  define void @test() gc "statepoint-example" {
    %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
    %safepoint_token1 = call token (i64, i32, void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
    ret void
  }

In this case, we've added an (unconditional) entry safepoint poll and converted the call into a ``gc.statepoint``.  Note that despite appearances, the entry poll is not necessarily redundant.  We'd have to know that ``foo`` and ``test`` were not mutually recursive for the poll to be redundant.  In practice, you'd probably want to your poll definition to contain a conditional branch of some form.


At the moment, PlaceSafepoints can insert safepoint polls at method entry and 
loop backedges locations.  Extending this to work with return polls would be 
straight forward if desired.

PlaceSafepoints includes a number of optimizations to avoid placing safepoint 
polls at particular sites unless needed to ensure timely execution of a poll 
under normal conditions.  PlaceSafepoints does not attempt to ensure timely 
execution of a poll under worst case conditions such as heavy system paging.

The implementation of a safepoint poll action is specified by looking up a 
function of the name ``gc.safepoint_poll`` in the containing Module.  The body
of this function is inserted at each poll site desired.  While calls or invokes
inside this method are transformed to a ``gc.statepoints``, recursive poll 
insertion is not performed.

By default PlaceSafepoints passes in ``0xABCDEF00`` as the statepoint
ID and ``0`` as the number of patchable bytes to the newly constructed
``gc.statepoint``.  These values can be configured on a per-callsite
basis using the attributes ``"statepoint-id"`` and
``"statepoint-num-patch-bytes"``.  If a call site is marked with a
``"statepoint-id"`` function attribute and its value is a positive
integer (represented as a string), then that value is used as the ID
of the newly constructed ``gc.statepoint``.  If a call site is marked
with a ``"statepoint-num-patch-bytes"`` function attribute and its
value is a positive integer, then that value is used as the 'num patch
bytes' parameter of the newly constructed ``gc.statepoint``.  The
``"statepoint-id"`` and ``"statepoint-num-patch-bytes"`` attributes
are not propagated to the ``gc.statepoint`` call or invoke if they
could be successfully parsed.

If you are scheduling the RewriteStatepointsForGC pass late in the pass order,
you should probably schedule this pass immediately before it.  The exception 
would be if you need to preserve abstract frame information (e.g. for
deoptimization or introspection) at safepoints.  In that case, ask on the 
llvm-dev mailing list for suggestions.


Supported Architectures
=======================

Support for statepoint generation requires some code for each backend.
Today, only X86_64 is supported.  

Problem Areas and Active Work
=============================

#. As the existing users of the late rewriting model have matured, we've found
   cases where the optimizer breaks the assumption that an SSA value of
   gc-pointer type actually contains a gc-pointer and vice-versa.  We need to
   clarify our expectations and propose at least one small IR change.  (Today,
   the gc-pointer distinction is managed via address spaces.  This turns out
   not to be quite strong enough.)

#. Support for languages which allow unmanaged pointers to garbage collected
   objects (i.e. pass a pointer to an object to a C routine) via pinning.

#. Support for garbage collected objects allocated on the stack.  Specifically,
   allocas are always assumed to be in address space 0 and we need a
   cast/promotion operator to let rewriting identify them.

#. The current statepoint lowering is known to be somewhat poor.  In the very
   long term, we'd like to integrate statepoints with the register allocator;
   in the near term this is unlikely to happen.  We've found the quality of
   lowering to be relatively unimportant as hot-statepoints are almost always
   inliner bugs.

#. Concerns have been raised that the statepoint representation results in a
   large amount of IR being produced for some examples and that this
   contributes to higher than expected memory usage and compile times.  There's
   no immediate plans to make changes due to this, but alternate models may be
   explored in the future.

#. Relocations along exceptional paths are currently broken in ToT.  In
   particular, there is current no way to represent a rethrow on a path which
   also has relocations.  See `this llvm-dev discussion
   <https://groups.google.com/forum/#!topic/llvm-dev/AE417XjgxvI>`_ for more
   detail.

Bugs and Enhancements
=====================

Currently known bugs and enhancements under consideration can be
tracked by performing a `bugzilla search
<http://llvm.org/bugs/buglist.cgi?cmdtype=runnamed&namedcmd=Statepoint%20Bugs&list_id=64342>`_
for [Statepoint] in the summary field. When filing new bugs, please
use this tag so that interested parties see the newly filed bug.  As
with most LLVM features, design discussions take place on `llvm-dev
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_, and patches
should be sent to `llvm-commits
<http://lists.llvm.org/mailman/listinfo/llvm-commits>`_ for review.

