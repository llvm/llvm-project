======================================
LLVM IR Undefined Behavior (UB) Manual
======================================

.. contents::
   :local:
   :depth: 2

Abstract
========
This document describes the undefined behavior (UB) in LLVM's IR, including
undef and poison values, as well as the ``freeze`` instruction.
We also provide guidelines on when to use each form of UB.


Introduction
============
Undefined behavior (UB) is used to specify the behavior of corner cases for
which we don't wish to specify the concrete results. UB is also used to provide
additional constraints to the optimizers (e.g., assumptions that the frontend
guarantees through the language type system or the runtime).
For example, we could specify the result of division by zero as zero, but
since we are not really interested in the result, we say it is UB.

There exist two forms of undefined behavior in LLVM: immediate UB and deferred
UB. The latter comes in two flavors: undef and poison values.
There is also a ``freeze`` instruction to tame the propagation of deferred UB.
The lattice of values in LLVM is:
immediate UB > poison > undef > freeze(poison) > concrete value.

We explain each of the concepts in detail below.


Immediate UB
============
Immediate UB is the most severe form of UB. It should be avoided whenever
possible.
Immediate UB should be used only for operations that trap in most CPUs supported
by LLVM.
Examples include division by zero, dereferencing a null pointer, etc.

The reason that immediate UB should be avoided is that it makes optimizations
such as hoisting a lot harder.
Consider the following example:

.. code-block:: llvm

    define i32 @f(i1 %c, i32 %v) {
      br i1 %c, label %then, label %else

    then:
      %div = udiv i32 3, %v
      br label %ret

    else:
      br label %ret

    ret:
      %r = phi i32 [ %div, %then ], [ 0, %else ]
      ret i32 %r
    }

We might be tempted to simplify this function by removing the branching and
executing the division speculatively because ``%c`` is true most of times.
We would obtain the following IR:

.. code-block:: llvm

    define i32 @f(i1 %c, i32 %v) {
      %div = udiv i32 3, %v
      %r = select i1 %c, i32 %div, i32 0
      ret i32 %r
    }

However, this transformation is not correct! Since division triggers UB
when the divisor is zero, we can only execute speculatively if we are sure we
don't hit that condition.
The function above, when called as ``f(false, 0)``, would return 0 before the
optimization, and triggers UB after being optimized.

This example highlights why we minimize the cases that trigger immediate UB
as much as possible.
As a rule of thumb, use immediate UB only for the cases that trap the CPU for
most of the supported architectures.


Time Travel
-----------
Immediate UB in LLVM IR allows the so-called time travelling. What this means
is that if a program triggers UB, then we are not required to preserve any of
its observable behavior, including I/O.
For example, the following function triggers UB after calling ``printf``:

.. code-block:: llvm

    define void @fn() {
      call void @printf(...) willreturn
      unreachable
    }

Since we know that ``printf`` will always return, and because LLVM's UB can
time-travel, it is legal to remove the call to ``printf`` altogether and
optimize the function to simply:

.. code-block:: llvm

    define void @fn() {
      unreachable
    }


Deferred UB
===========
Deferred UB is a lighter form of UB. It enables instructions to be executed
speculatively while marking some corner cases as having erroneous values.
Deferred UB should be used for cases where the semantics offered by common
CPUs differ, but the CPU does not trap.

As an example, consider the shift instructions. The x86 and ARM architectures
offer different semantics when the shift amount is equal to or greater than
the bitwidth.
We could solve this tension in one of two ways: 1) pick one of the x86/ARM
semantics for LLVM, which would make the code emitted for the other architecture
slower; 2) define that case as yielding ``poison``.
LLVM chose the latter option. For frontends for languages like C or C++
(e.g., clang), they can map shifts in the source program directly to a shift in
LLVM IR, since the semantics of C and C++ define such shifts as UB.
For languages that offer strong semantics, they must use the value of the shift
conditionally, e.g.:

.. code-block:: llvm

    define i32 @x86_shift(i32 %a, i32 %b) {
      %mask = and i32 %b, 31
      %shift = shl i32 %a, %mask
      ret i32 %shift
    }


There are two deferred UB values in LLVM: ``undef`` and ``poison``, which we
describe next.


Undef Values
------------
.. warning::
   Undef values are deprecated and should be used only when strictly necessary.
   Uses of undef values should be restricted to representing loads of
   uninitialized memory. This is the only part of the IR semantics that cannot
   be replaced with alternatives yet (work in ongoing).

An undef value represents any value of a given type. Moreover, each use of
an instruction that depends on undef can observe a different value.
For example:

.. code-block:: llvm

    define i32 @fn() {
      %add = add i32 undef, 0
      %ret = add i32 %add, %add
      ret i32 %ret
    }

Unsurprisingly, the first addition yields ``undef``.
However, the result of the second addition is more subtle. We might be tempted
to think that it yields an even number. But it might not be!
Since each (transitive) use of ``undef`` can observe a different value,
the second addition is equivalent to ``add i32 undef, undef``, which is
equivalent to ``undef``.
Hence, the function above is equivalent to:

.. code-block:: llvm

    define i32 @fn() {
      ret i32 undef
    }

Each call to this function may observe a different value, namely any 32-bit
number (even and odd).

Because each use of undef can observe a different value, some optimizations
are wrong if we are not sure a value is not undef.
Consider a function that multiplies a number by 2:

.. code-block:: llvm

    define i32 @fn(i32 %v) {
      %mul2 = mul i32 %v, 2
      ret i32 %mul2
    }

This function is guaranteed to return an even number, even if ``%v`` is
undef.
However, as we've seen above, the following function does not:

.. code-block:: llvm

    define i32 @fn(i32 %v) {
      %mul2 = add i32 %v, %v
      ret i32 %mul2
    }

This optimization is wrong just because undef values exist, even if they are
not used in this part of the program as LLVM has no way to tell if ``%v`` is
undef or not.

Looking at the value lattice, ``undef`` values can only be replaced with either
a ``freeze`` instruction or a concrete value.
A consequence is that giving undef as an operand to an instruction that triggers
UB for some values of that operand makes the program UB. For example,
``udiv %x, undef`` is UB since we replace undef with 0 (``udiv %x, 0``),
becoming obvious that it is UB.


Poison Values
-------------
Poison values are a stronger form of deferred UB than undef. They still
allow instructions to be executed speculatively, but they taint the whole
expression DAG (with some exceptions), akin to floating point NaN values.

Example:

.. code-block:: llvm

    define i32 @fn(i32 %a, i32 %b, i32 %c) {
      %add = add nsw i32 %a, %b
      %ret = add nsw i32 %add, %c
      ret i32 %ret
    }

The ``nsw`` attribute in the additions indicates that the operation yields
poison if there is a signed overflow.
If the first addition overflows, ``%add`` is poison and thus ``%ret`` is also
poison since it taints the whole expression DAG.

Poison values can be replaced with any value of type (undef, concrete values,
or a ``freeze`` instruction).


Propagation of Poison Through Select
------------------------------------
Most instructions return poison if any of their inputs is poison.
A notable exception is the ``select`` instruction, which is poison if and
only if the condition is poison or the selected value is poison.
This means that ``select`` acts as a barrier for poison propagation, which
impacts which optimizations can be performed.

For example, consider the following function:

.. code-block:: llvm

  define i1 @fn(i32 %x, i32 %y) {
    %cmp1 = icmp ne i32 %x, 0
    %cmp2 = icmp ugt i32 %x, %y
    %and = select i1 %cmp1, i1 %cmp2, i1 false
    ret i1 %and
  }

It is not correct to optimize the ``select`` into an ``and`` because when
``%cmp1`` is false, the ``select`` is only poison if ``%x`` is poison, while
the ``and`` below is poison if either ``%x`` or ``%y`` are poison.

.. code-block:: llvm

  define i1 @fn(i32 %x, i32 %y) {
    %cmp1 = icmp ne i32 %x, 0
    %cmp2 = icmp ugt i32 %x, %y
    %and = and i1 %cmp1, %cmp2     ;; poison if %x or %y are poison
    ret i1 %and
  }

However, the optimization is possible if all operands of the values are used in
the condition (notice the flipped operands in the ``select``):

.. code-block:: llvm

  define i1 @fn(i32 %x, i32 %y) {
    %cmp1 = icmp ne i32 %x, 0
    %cmp2 = icmp ugt i32 %x, %y
    %and = select i1 %cmp2, i1 %cmp1, i1 false
    ; ok to replace with:
    %and = and i1 %cmp1, %cmp2
    ret i1 %and
  }


The Freeze Instruction
======================
Both undef and poison values sometimes propagate too much down an expression
DAG. Undef values because each transitive use can observe a different value,
and poison values because they make the whole DAG poison.
There are some cases where it is important to stop such propagation.
This is where the ``freeze`` instruction comes in.

Take the following example function:

.. code-block:: llvm

    define i32 @fn(i32 %n, i1 %c) {
    entry:
      br label %loop

   loop:
      %i = phi i32 [ 0, %entry ], [ %i2, %loop.end ]
      %cond = icmp ule i32 %i, %n
      br i1 %cond, label %loop.cont, label %exit

   loop.cont:
      br i1 %c, label %then, label %else

    then:
      ...
      br label %loop.end

    else:
      ...
      br label %loop.end

    loop.end:
      %i2 = add i32 %i, 1
      br label %loop

    exit:
      ...
    }

Imagine we want to perform loop unswitching on the loop above since the branch
condition inside the loop is loop invariant.
We would obtain the following IR:

.. code-block:: llvm

    define i32 @fn(i32 %n, i1 %c) {
    entry:
      br i1 %c, label %then, label %else

   then:
      %i = phi i32 [ 0, %entry ], [ %i2, %then.cont ]
      %cond = icmp ule i32 %i, %n
      br i1 %cond, label %then.cont, label %exit

   then.cont:
      ...
      %i2 = add i32 %i, 1
      br label %then

   else:
      %i3 = phi i32 [ 0, %entry ], [ %i4, %else.cont ]
      %cond = icmp ule i32 %i3, %n
      br i1 %cond, label %else.cont, label %exit

   else.cont:
      ...
      %i4 = add i32 %i3, 1
      br label %else

    exit:
      ...
    }

There is a subtle catch: when the function is called with ``%n`` being zero,
the original function did not branch on ``%c``, while the optimized one does.
Branching on a deferred UB value is immediate UB, hence the transformation is
wrong in general because ``%c`` may be undef or poison.

Cases like this need a way to tame deferred UB values. This is exactly what the
``freeze`` instruction is for!
When given a concrete value as argument, ``freeze`` is a no-op, returning the
argument as-is. When given an undef or poison value, ``freeze`` returns a
non-deterministic value of the type.
This is not the same as undef: the value returned by ``freeze`` is the same
for all users.

Branching on a value returned by ``freeze`` is always safe since it either
evaluates to true or false consistently.
We can make the loop unswitching optimization above correct as follows:

.. code-block:: llvm

    define i32 @fn(i32 %n, i1 %c) {
    entry:
      %c2 = freeze i1 %c
      br i1 %c2, label %then, label %else


Writing Tests Without Undefined Behavior
========================================

When writing tests, it is important to ensure that they don't trigger UB
unnecessarily. Some automated test reduces sometimes use undef or poison
values as dummy values, but this is considered a bad practice if this leads
to triggering UB.

For example, imagine that we want to write a test and we don't care about the
particular divisor value because our optimization kicks in regardless:

.. code-block:: llvm

    define i32 @fn(i8 %a) {
      %div = udiv i8 %a, poison
      ...
   }

The issue with this test is that it triggers immediate UB. This prevents
verification tools like Alive from validating the correctness of the
optimization. Hence, it is considered a bad practice to have tests with
unnecessary immediate UB (unless that is exactly what the test is for).
The test above should use a dummy function argument instead of using poison:

.. code-block:: llvm

    define i32 @fn(i8 %a, i8 %dummy) {
      %div = udiv i8 %a, %dummy
      ...
   }

Common sources of immediate UB in tests include branching on undef/poison
conditions and dereferencing undef/poison/null pointers.

.. note::
   If you need a placeholder value to pass as an argument to an instruction
   that may trigger UB, add a new argument to the function rather than using
   undef or poison.


Summary
=======
Undefined behavior (UB) in LLVM IR consists of two well-defined concepts:
immediate and deferred UB (undef and poison values).
Passing deferred UB values to certain operations leads to immediate UB.
This can be avoided in some cases through the use of the ``freeze``
instruction.

The lattice of values in LLVM is:
immediate UB > poison > undef > freeze(poison) > concrete value.
It is only valid to transform values from the left to the right (e.g., a poison
value can be replaced with a concrete value, but not the other way around).

Undef is now deprecated and should be used only to represent loads of
uninitialized memory.
