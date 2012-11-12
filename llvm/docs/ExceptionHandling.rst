.. _exception_handling:

==========================
Exception Handling in LLVM
==========================

.. contents::
   :local:

Introduction
============

This document is the central repository for all information pertaining to
exception handling in LLVM.  It describes the format that LLVM exception
handling information takes, which is useful for those interested in creating
front-ends or dealing directly with the information.  Further, this document
provides specific examples of what exception handling information is used for in
C and C++.

Itanium ABI Zero-cost Exception Handling
----------------------------------------

Exception handling for most programming languages is designed to recover from
conditions that rarely occur during general use of an application.  To that end,
exception handling should not interfere with the main flow of an application's
algorithm by performing checkpointing tasks, such as saving the current pc or
register state.

The Itanium ABI Exception Handling Specification defines a methodology for
providing outlying data in the form of exception tables without inlining
speculative exception handling code in the flow of an application's main
algorithm.  Thus, the specification is said to add "zero-cost" to the normal
execution of an application.

A more complete description of the Itanium ABI exception handling runtime
support of can be found at `Itanium C++ ABI: Exception Handling
<http://www.codesourcery.com/cxx-abi/abi-eh.html>`_. A description of the
exception frame format can be found at `Exception Frames
<http://refspecs.freestandards.org/LSB_3.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html>`_,
with details of the DWARF 4 specification at `DWARF 4 Standard
<http://dwarfstd.org/Dwarf4Std.php>`_.  A description for the C++ exception
table formats can be found at `Exception Handling Tables
<http://www.codesourcery.com/cxx-abi/exceptions.pdf>`_.

Setjmp/Longjmp Exception Handling
---------------------------------

Setjmp/Longjmp (SJLJ) based exception handling uses LLVM intrinsics
`llvm.eh.sjlj.setjmp`_ and `llvm.eh.sjlj.longjmp`_ to handle control flow for
exception handling.

For each function which does exception processing --- be it ``try``/``catch``
blocks or cleanups --- that function registers itself on a global frame
list. When exceptions are unwinding, the runtime uses this list to identify
which functions need processing.

Landing pad selection is encoded in the call site entry of the function
context. The runtime returns to the function via `llvm.eh.sjlj.longjmp`_, where
a switch table transfers control to the appropriate landing pad based on the
index stored in the function context.

In contrast to DWARF exception handling, which encodes exception regions and
frame information in out-of-line tables, SJLJ exception handling builds and
removes the unwind frame context at runtime. This results in faster exception
handling at the expense of slower execution when no exceptions are thrown. As
exceptions are, by their nature, intended for uncommon code paths, DWARF
exception handling is generally preferred to SJLJ.

Overview
--------

When an exception is thrown in LLVM code, the runtime does its best to find a
handler suited to processing the circumstance.

The runtime first attempts to find an *exception frame* corresponding to the
function where the exception was thrown.  If the programming language supports
exception handling (e.g. C++), the exception frame contains a reference to an
exception table describing how to process the exception.  If the language does
not support exception handling (e.g. C), or if the exception needs to be
forwarded to a prior activation, the exception frame contains information about
how to unwind the current activation and restore the state of the prior
activation.  This process is repeated until the exception is handled. If the
exception is not handled and no activations remain, then the application is
terminated with an appropriate error message.

Because different programming languages have different behaviors when handling
exceptions, the exception handling ABI provides a mechanism for
supplying *personalities*. An exception handling personality is defined by
way of a *personality function* (e.g. ``__gxx_personality_v0`` in C++),
which receives the context of the exception, an *exception structure*
containing the exception object type and value, and a reference to the exception
table for the current function.  The personality function for the current
compile unit is specified in a *common exception frame*.

The organization of an exception table is language dependent. For C++, an
exception table is organized as a series of code ranges defining what to do if
an exception occurs in that range. Typically, the information associated with a
range defines which types of exception objects (using C++ *type info*) that are
handled in that range, and an associated action that should take place. Actions
typically pass control to a *landing pad*.

A landing pad corresponds roughly to the code found in the ``catch`` portion of
a ``try``/``catch`` sequence. When execution resumes at a landing pad, it
receives an *exception structure* and a *selector value* corresponding to the
*type* of exception thrown. The selector is then used to determine which *catch*
should actually process the exception.

LLVM Code Generation
====================

From a C++ developer's perspective, exceptions are defined in terms of the
``throw`` and ``try``/``catch`` statements. In this section we will describe the
implementation of LLVM exception handling in terms of C++ examples.

Throw
-----

Languages that support exception handling typically provide a ``throw``
operation to initiate the exception process. Internally, a ``throw`` operation
breaks down into two steps.

#. A request is made to allocate exception space for an exception structure.
   This structure needs to survive beyond the current activation. This structure
   will contain the type and value of the object being thrown.

#. A call is made to the runtime to raise the exception, passing the exception
   structure as an argument.

In C++, the allocation of the exception structure is done by the
``__cxa_allocate_exception`` runtime function. The exception raising is handled
by ``__cxa_throw``. The type of the exception is represented using a C++ RTTI
structure.

Try/Catch
---------

A call within the scope of a *try* statement can potentially raise an
exception. In those circumstances, the LLVM C++ front-end replaces the call with
an ``invoke`` instruction. Unlike a call, the ``invoke`` has two potential
continuation points:

#. where to continue when the call succeeds as per normal, and

#. where to continue if the call raises an exception, either by a throw or the
   unwinding of a throw

The term used to define a the place where an ``invoke`` continues after an
exception is called a *landing pad*. LLVM landing pads are conceptually
alternative function entry points where an exception structure reference and a
type info index are passed in as arguments. The landing pad saves the exception
structure reference and then proceeds to select the catch block that corresponds
to the type info of the exception object.

The LLVM `landingpad instruction <LangRef.html#i_landingpad>`_ is used to convey
information about the landing pad to the back end. For C++, the ``landingpad``
instruction returns a pointer and integer pair corresponding to the pointer to
the *exception structure* and the *selector value* respectively.

The ``landingpad`` instruction takes a reference to the personality function to
be used for this ``try``/``catch`` sequence. The remainder of the instruction is
a list of *cleanup*, *catch*, and *filter* clauses. The exception is tested
against the clauses sequentially from first to last. The selector value is a
positive number if the exception matched a type info, a negative number if it
matched a filter, and zero if it matched a cleanup. If nothing is matched, the
behavior of the program is `undefined`_. If a type info matched, then the
selector value is the index of the type info in the exception table, which can
be obtained using the `llvm.eh.typeid.for`_ intrinsic.

Once the landing pad has the type info selector, the code branches to the code
for the first catch. The catch then checks the value of the type info selector
against the index of type info for that catch.  Since the type info index is not
known until all the type infos have been gathered in the backend, the catch code
must call the `llvm.eh.typeid.for`_ intrinsic to determine the index for a given
type info. If the catch fails to match the selector then control is passed on to
the next catch.

Finally, the entry and exit of catch code is bracketed with calls to
``__cxa_begin_catch`` and ``__cxa_end_catch``.

* ``__cxa_begin_catch`` takes an exception structure reference as an argument
  and returns the value of the exception object.

* ``__cxa_end_catch`` takes no arguments. This function:

  #. Locates the most recently caught exception and decrements its handler
     count,

  #. Removes the exception from the *caught* stack if the handler count goes to
     zero, and

  #. Destroys the exception if the handler count goes to zero and the exception
     was not re-thrown by throw.

  .. note::

    a rethrow from within the catch may replace this call with a
    ``__cxa_rethrow``.

Cleanups
--------

A cleanup is extra code which needs to be run as part of unwinding a scope.  C++
destructors are a typical example, but other languages and language extensions
provide a variety of different kinds of cleanups. In general, a landing pad may
need to run arbitrary amounts of cleanup code before actually entering a catch
block. To indicate the presence of cleanups, a `landingpad
instruction <LangRef.html#i_landingpad>`_ should have a *cleanup*
clause. Otherwise, the unwinder will not stop at the landing pad if there are no
catches or filters that require it to.

.. note::

  Do not allow a new exception to propagate out of the execution of a
  cleanup. This can corrupt the internal state of the unwinder.  Different
  languages describe different high-level semantics for these situations: for
  example, C++ requires that the process be terminated, whereas Ada cancels both
  exceptions and throws a third.

When all cleanups are finished, if the exception is not handled by the current
function, resume unwinding by calling the `resume
instruction <LangRef.html#i_resume>`_, passing in the result of the
``landingpad`` instruction for the original landing pad.

Throw Filters
-------------

C++ allows the specification of which exception types may be thrown from a
function. To represent this, a top level landing pad may exist to filter out
invalid types. To express this in LLVM code the `landingpad
instruction <LangRef.html#i_landingpad>`_ will have a filter clause. The clause
consists of an array of type infos.  ``landingpad`` will return a negative value
if the exception does not match any of the type infos. If no match is found then
a call to ``__cxa_call_unexpected`` should be made, otherwise
``_Unwind_Resume``.  Each of these functions requires a reference to the
exception structure.  Note that the most general form of a ``landingpad``
instruction can have any number of catch, cleanup, and filter clauses (though
having more than one cleanup is pointless). The LLVM C++ front-end can generate
such ``landingpad`` instructions due to inlining creating nested exception
handling scopes.

.. _undefined:

Restrictions
------------

The unwinder delegates the decision of whether to stop in a call frame to that
call frame's language-specific personality function. Not all unwinders guarantee
that they will stop to perform cleanups. For example, the GNU C++ unwinder
doesn't do so unless the exception is actually caught somewhere further up the
stack.

In order for inlining to behave correctly, landing pads must be prepared to
handle selector results that they did not originally advertise. Suppose that a
function catches exceptions of type ``A``, and it's inlined into a function that
catches exceptions of type ``B``. The inliner will update the ``landingpad``
instruction for the inlined landing pad to include the fact that ``B`` is also
caught. If that landing pad assumes that it will only be entered to catch an
``A``, it's in for a rude awakening.  Consequently, landing pads must test for
the selector results they understand and then resume exception propagation with
the `resume instruction <LangRef.html#i_resume>`_ if none of the conditions
match.

Exception Handling Intrinsics
=============================

In addition to the ``landingpad`` and ``resume`` instructions, LLVM uses several
intrinsic functions (name prefixed with ``llvm.eh``) to provide exception
handling information at various points in generated code.

.. _llvm.eh.typeid.for:

llvm.eh.typeid.for
------------------

.. code-block:: llvm

  i32 @llvm.eh.typeid.for(i8* %type_info)


This intrinsic returns the type info index in the exception table of the current
function.  This value can be used to compare against the result of
``landingpad`` instruction.  The single argument is a reference to a type info.

.. _llvm.eh.sjlj.setjmp:

llvm.eh.sjlj.setjmp
-------------------

.. code-block:: llvm

  i32 @llvm.eh.sjlj.setjmp(i8* %setjmp_buf)

For SJLJ based exception handling, this intrinsic forces register saving for the
current function and stores the address of the following instruction for use as
a destination address by `llvm.eh.sjlj.longjmp`_. The buffer format and the
overall functioning of this intrinsic is compatible with the GCC
``__builtin_setjmp`` implementation allowing code built with the clang and GCC
to interoperate.

The single parameter is a pointer to a five word buffer in which the calling
context is saved. The front end places the frame pointer in the first word, and
the target implementation of this intrinsic should place the destination address
for a `llvm.eh.sjlj.longjmp`_ in the second word. The following three words are
available for use in a target-specific manner.

.. _llvm.eh.sjlj.longjmp:

llvm.eh.sjlj.longjmp
--------------------

.. code-block:: llvm

  void @llvm.eh.sjlj.longjmp(i8* %setjmp_buf)

For SJLJ based exception handling, the ``llvm.eh.sjlj.longjmp`` intrinsic is
used to implement ``__builtin_longjmp()``. The single parameter is a pointer to
a buffer populated by `llvm.eh.sjlj.setjmp`_. The frame pointer and stack
pointer are restored from the buffer, then control is transferred to the
destination address.

llvm.eh.sjlj.lsda
-----------------

.. code-block:: llvm

  i8* @llvm.eh.sjlj.lsda()

For SJLJ based exception handling, the ``llvm.eh.sjlj.lsda`` intrinsic returns
the address of the Language Specific Data Area (LSDA) for the current
function. The SJLJ front-end code stores this address in the exception handling
function context for use by the runtime.

llvm.eh.sjlj.callsite
---------------------

.. code-block:: llvm

  void @llvm.eh.sjlj.callsite(i32 %call_site_num)

For SJLJ based exception handling, the ``llvm.eh.sjlj.callsite`` intrinsic
identifies the callsite value associated with the following ``invoke``
instruction. This is used to ensure that landing pad entries in the LSDA are
generated in matching order.

Asm Table Formats
=================

There are two tables that are used by the exception handling runtime to
determine which actions should be taken when an exception is thrown.

Exception Handling Frame
------------------------

An exception handling frame ``eh_frame`` is very similar to the unwind frame
used by DWARF debug info. The frame contains all the information necessary to
tear down the current frame and restore the state of the prior frame. There is
an exception handling frame for each function in a compile unit, plus a common
exception handling frame that defines information common to all functions in the
unit.

Exception Tables
----------------

An exception table contains information about what actions to take when an
exception is thrown in a particular part of a function's code. There is one
exception table per function, except leaf functions and functions that have
calls only to non-throwing functions. They do not need an exception table.
