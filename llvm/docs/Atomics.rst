.. _atomics:

==============================================
LLVM Atomic Instructions and Concurrency Guide
==============================================

.. contents::
   :local:

Introduction
============

Historically, LLVM has not had very strong support for concurrency; some minimal
intrinsics were provided, and ``volatile`` was used in some cases to achieve
rough semantics in the presence of concurrency.  However, this is changing;
there are now new instructions which are well-defined in the presence of threads
and asynchronous signals, and the model for existing instructions has been
clarified in the IR.

The atomic instructions are designed specifically to provide readable IR and
optimized code generation for the following:

* The new C++0x ``<atomic>`` header.  (`C++0x draft available here
  <http://www.open-std.org/jtc1/sc22/wg21/>`_.) (`C1x draft available here
  <http://www.open-std.org/jtc1/sc22/wg14/>`_.)

* Proper semantics for Java-style memory, for both ``volatile`` and regular
  shared variables. (`Java Specification
  <http://java.sun.com/docs/books/jls/third_edition/html/memory.html>`_)

* gcc-compatible ``__sync_*`` builtins. (`Description
  <http://gcc.gnu.org/onlinedocs/gcc/Atomic-Builtins.html>`_)

* Other scenarios with atomic semantics, including ``static`` variables with
  non-trivial constructors in C++.

Atomic and volatile in the IR are orthogonal; "volatile" is the C/C++ volatile,
which ensures that every volatile load and store happens and is performed in the
stated order.  A couple examples: if a SequentiallyConsistent store is
immediately followed by another SequentiallyConsistent store to the same
address, the first store can be erased. This transformation is not allowed for a
pair of volatile stores. On the other hand, a non-volatile non-atomic load can
be moved across a volatile load freely, but not an Acquire load.

This document is intended to provide a guide to anyone either writing a frontend
for LLVM or working on optimization passes for LLVM with a guide for how to deal
with instructions with special semantics in the presence of concurrency.  This
is not intended to be a precise guide to the semantics; the details can get
extremely complicated and unreadable, and are not usually necessary.

.. _Optimization outside atomic:

Optimization outside atomic
===========================

The basic ``'load'`` and ``'store'`` allow a variety of optimizations, but can
lead to undefined results in a concurrent environment; see `NotAtomic`_. This
section specifically goes into the one optimizer restriction which applies in
concurrent environments, which gets a bit more of an extended description
because any optimization dealing with stores needs to be aware of it.

From the optimizer's point of view, the rule is that if there are not any
instructions with atomic ordering involved, concurrency does not matter, with
one exception: if a variable might be visible to another thread or signal
handler, a store cannot be inserted along a path where it might not execute
otherwise.  Take the following example:

.. code-block:: c

 /* C code, for readability; run through clang -O2 -S -emit-llvm to get
     equivalent IR */
  int x;
  void f(int* a) {
    for (int i = 0; i < 100; i++) {
      if (a[i])
        x += 1;
    }
  }

The following is equivalent in non-concurrent situations:

.. code-block:: c

  int x;
  void f(int* a) {
    int xtemp = x;
    for (int i = 0; i < 100; i++) {
      if (a[i])
        xtemp += 1;
    }
    x = xtemp;
  }

However, LLVM is not allowed to transform the former to the latter: it could
indirectly introduce undefined behavior if another thread can access ``x`` at
the same time. (This example is particularly of interest because before the
concurrency model was implemented, LLVM would perform this transformation.)

Note that speculative loads are allowed; a load which is part of a race returns
``undef``, but does not have undefined behavior.

Atomic instructions
===================

For cases where simple loads and stores are not sufficient, LLVM provides
various atomic instructions. The exact guarantees provided depend on the
ordering; see `Atomic orderings`_.

``load atomic`` and ``store atomic`` provide the same basic functionality as
non-atomic loads and stores, but provide additional guarantees in situations
where threads and signals are involved.

``cmpxchg`` and ``atomicrmw`` are essentially like an atomic load followed by an
atomic store (where the store is conditional for ``cmpxchg``), but no other
memory operation can happen on any thread between the load and store.  Note that
LLVM's cmpxchg does not provide quite as many options as the C++0x version.

A ``fence`` provides Acquire and/or Release ordering which is not part of
another operation; it is normally used along with Monotonic memory operations.
A Monotonic load followed by an Acquire fence is roughly equivalent to an
Acquire load.

Frontends generating atomic instructions generally need to be aware of the
target to some degree; atomic instructions are guaranteed to be lock-free, and
therefore an instruction which is wider than the target natively supports can be
impossible to generate.

.. _Atomic orderings:

Atomic orderings
================

In order to achieve a balance between performance and necessary guarantees,
there are six levels of atomicity. They are listed in order of strength; each
level includes all the guarantees of the previous level except for
Acquire/Release. (See also `LangRef Ordering <LangRef.html#ordering>`_.)

.. _NotAtomic:

NotAtomic
---------

NotAtomic is the obvious, a load or store which is not atomic. (This isn't
really a level of atomicity, but is listed here for comparison.) This is
essentially a regular load or store. If there is a race on a given memory
location, loads from that location return undef.

Relevant standard
  This is intended to match shared variables in C/C++, and to be used in any
  other context where memory access is necessary, and a race is impossible. (The
  precise definition is in `LangRef Memory Model <LangRef.html#memmodel>`_.)

Notes for frontends
  The rule is essentially that all memory accessed with basic loads and stores
  by multiple threads should be protected by a lock or other synchronization;
  otherwise, you are likely to run into undefined behavior. If your frontend is
  for a "safe" language like Java, use Unordered to load and store any shared
  variable.  Note that NotAtomic volatile loads and stores are not properly
  atomic; do not try to use them as a substitute. (Per the C/C++ standards,
  volatile does provide some limited guarantees around asynchronous signals, but
  atomics are generally a better solution.)

Notes for optimizers
  Introducing loads to shared variables along a codepath where they would not
  otherwise exist is allowed; introducing stores to shared variables is not. See
  `Optimization outside atomic`_.

Notes for code generation
  The one interesting restriction here is that it is not allowed to write to
  bytes outside of the bytes relevant to a store.  This is mostly relevant to
  unaligned stores: it is not allowed in general to convert an unaligned store
  into two aligned stores of the same width as the unaligned store. Backends are
  also expected to generate an i8 store as an i8 store, and not an instruction
  which writes to surrounding bytes.  (If you are writing a backend for an
  architecture which cannot satisfy these restrictions and cares about
  concurrency, please send an email to llvmdev.)

Unordered
---------

Unordered is the lowest level of atomicity. It essentially guarantees that races
produce somewhat sane results instead of having undefined behavior.  It also
guarantees the operation to be lock-free, so it do not depend on the data being
part of a special atomic structure or depend on a separate per-process global
lock.  Note that code generation will fail for unsupported atomic operations; if
you need such an operation, use explicit locking.

Relevant standard
  This is intended to match the Java memory model for shared variables.

Notes for frontends
  This cannot be used for synchronization, but is useful for Java and other
  "safe" languages which need to guarantee that the generated code never
  exhibits undefined behavior. Note that this guarantee is cheap on common
  platforms for loads of a native width, but can be expensive or unavailable for
  wider loads, like a 64-bit store on ARM. (A frontend for Java or other "safe"
  languages would normally split a 64-bit store on ARM into two 32-bit unordered
  stores.)

Notes for optimizers
  In terms of the optimizer, this prohibits any transformation that transforms a
  single load into multiple loads, transforms a store into multiple stores,
  narrows a store, or stores a value which would not be stored otherwise.  Some
  examples of unsafe optimizations are narrowing an assignment into a bitfield,
  rematerializing a load, and turning loads and stores into a memcpy
  call. Reordering unordered operations is safe, though, and optimizers should
  take advantage of that because unordered operations are common in languages
  that need them.

Notes for code generation
  These operations are required to be atomic in the sense that if you use
  unordered loads and unordered stores, a load cannot see a value which was
  never stored.  A normal load or store instruction is usually sufficient, but
  note that an unordered load or store cannot be split into multiple
  instructions (or an instruction which does multiple memory operations, like
  ``LDRD`` on ARM).

Monotonic
---------

Monotonic is the weakest level of atomicity that can be used in synchronization
primitives, although it does not provide any general synchronization. It
essentially guarantees that if you take all the operations affecting a specific
address, a consistent ordering exists.

Relevant standard
  This corresponds to the C++0x/C1x ``memory_order_relaxed``; see those
  standards for the exact definition.

Notes for frontends
  If you are writing a frontend which uses this directly, use with caution.  The
  guarantees in terms of synchronization are very weak, so make sure these are
  only used in a pattern which you know is correct.  Generally, these would
  either be used for atomic operations which do not protect other memory (like
  an atomic counter), or along with a ``fence``.

Notes for optimizers
  In terms of the optimizer, this can be treated as a read+write on the relevant
  memory location (and alias analysis will take advantage of that). In addition,
  it is legal to reorder non-atomic and Unordered loads around Monotonic
  loads. CSE/DSE and a few other optimizations are allowed, but Monotonic
  operations are unlikely to be used in ways which would make those
  optimizations useful.

Notes for code generation
  Code generation is essentially the same as that for unordered for loads and
  stores.  No fences are required.  ``cmpxchg`` and ``atomicrmw`` are required
  to appear as a single operation.

Acquire
-------

Acquire provides a barrier of the sort necessary to acquire a lock to access
other memory with normal loads and stores.

Relevant standard
  This corresponds to the C++0x/C1x ``memory_order_acquire``. It should also be
  used for C++0x/C1x ``memory_order_consume``.

Notes for frontends
  If you are writing a frontend which uses this directly, use with caution.
  Acquire only provides a semantic guarantee when paired with a Release
  operation.

Notes for optimizers
  Optimizers not aware of atomics can treat this like a nothrow call.  It is
  also possible to move stores from before an Acquire load or read-modify-write
  operation to after it, and move non-Acquire loads from before an Acquire
  operation to after it.

Notes for code generation
  Architectures with weak memory ordering (essentially everything relevant today
  except x86 and SPARC) require some sort of fence to maintain the Acquire
  semantics.  The precise fences required varies widely by architecture, but for
  a simple implementation, most architectures provide a barrier which is strong
  enough for everything (``dmb`` on ARM, ``sync`` on PowerPC, etc.).  Putting
  such a fence after the equivalent Monotonic operation is sufficient to
  maintain Acquire semantics for a memory operation.

Release
-------

Release is similar to Acquire, but with a barrier of the sort necessary to
release a lock.

Relevant standard
  This corresponds to the C++0x/C1x ``memory_order_release``.

Notes for frontends
  If you are writing a frontend which uses this directly, use with caution.
  Release only provides a semantic guarantee when paired with a Acquire
  operation.

Notes for optimizers
  Optimizers not aware of atomics can treat this like a nothrow call.  It is
  also possible to move loads from after a Release store or read-modify-write
  operation to before it, and move non-Release stores from after an Release
  operation to before it.

Notes for code generation
  See the section on Acquire; a fence before the relevant operation is usually
  sufficient for Release. Note that a store-store fence is not sufficient to
  implement Release semantics; store-store fences are generally not exposed to
  IR because they are extremely difficult to use correctly.

AcquireRelease
--------------

AcquireRelease (``acq_rel`` in IR) provides both an Acquire and a Release
barrier (for fences and operations which both read and write memory).

Relevant standard
  This corresponds to the C++0x/C1x ``memory_order_acq_rel``.

Notes for frontends
  If you are writing a frontend which uses this directly, use with caution.
  Acquire only provides a semantic guarantee when paired with a Release
  operation, and vice versa.

Notes for optimizers
  In general, optimizers should treat this like a nothrow call; the possible
  optimizations are usually not interesting.

Notes for code generation
  This operation has Acquire and Release semantics; see the sections on Acquire
  and Release.

SequentiallyConsistent
----------------------

SequentiallyConsistent (``seq_cst`` in IR) provides Acquire semantics for loads
and Release semantics for stores. Additionally, it guarantees that a total
ordering exists between all SequentiallyConsistent operations.

Relevant standard
  This corresponds to the C++0x/C1x ``memory_order_seq_cst``, Java volatile, and
  the gcc-compatible ``__sync_*`` builtins which do not specify otherwise.

Notes for frontends
  If a frontend is exposing atomic operations, these are much easier to reason
  about for the programmer than other kinds of operations, and using them is
  generally a practical performance tradeoff.

Notes for optimizers
  Optimizers not aware of atomics can treat this like a nothrow call.  For
  SequentiallyConsistent loads and stores, the same reorderings are allowed as
  for Acquire loads and Release stores, except that SequentiallyConsistent
  operations may not be reordered.

Notes for code generation
  SequentiallyConsistent loads minimally require the same barriers as Acquire
  operations and SequentiallyConsistent stores require Release
  barriers. Additionally, the code generator must enforce ordering between
  SequentiallyConsistent stores followed by SequentiallyConsistent loads. This
  is usually done by emitting either a full fence before the loads or a full
  fence after the stores; which is preferred varies by architecture.

Atomics and IR optimization
===========================

Predicates for optimizer writers to query:

* ``isSimple()``: A load or store which is not volatile or atomic.  This is
  what, for example, memcpyopt would check for operations it might transform.

* ``isUnordered()``: A load or store which is not volatile and at most
  Unordered. This would be checked, for example, by LICM before hoisting an
  operation.

* ``mayReadFromMemory()``/``mayWriteToMemory()``: Existing predicate, but note
  that they return true for any operation which is volatile or at least
  Monotonic.

* Alias analysis: Note that AA will return ModRef for anything Acquire or
  Release, and for the address accessed by any Monotonic operation.

To support optimizing around atomic operations, make sure you are using the
right predicates; everything should work if that is done.  If your pass should
optimize some atomic operations (Unordered operations in particular), make sure
it doesn't replace an atomic load or store with a non-atomic operation.

Some examples of how optimizations interact with various kinds of atomic
operations:

* ``memcpyopt``: An atomic operation cannot be optimized into part of a
  memcpy/memset, including unordered loads/stores.  It can pull operations
  across some atomic operations.

* LICM: Unordered loads/stores can be moved out of a loop.  It just treats
  monotonic operations like a read+write to a memory location, and anything
  stricter than that like a nothrow call.

* DSE: Unordered stores can be DSE'ed like normal stores.  Monotonic stores can
  be DSE'ed in some cases, but it's tricky to reason about, and not especially
  important.

* Folding a load: Any atomic load from a constant global can be constant-folded,
  because it cannot be observed.  Similar reasoning allows scalarrepl with
  atomic loads and stores.

Atomics and Codegen
===================

Atomic operations are represented in the SelectionDAG with ``ATOMIC_*`` opcodes.
On architectures which use barrier instructions for all atomic ordering (like
ARM), appropriate fences are split out as the DAG is built.

The MachineMemOperand for all atomic operations is currently marked as volatile;
this is not correct in the IR sense of volatile, but CodeGen handles anything
marked volatile very conservatively.  This should get fixed at some point.

Common architectures have some way of representing at least a pointer-sized
lock-free ``cmpxchg``; such an operation can be used to implement all the other
atomic operations which can be represented in IR up to that size.  Backends are
expected to implement all those operations, but not operations which cannot be
implemented in a lock-free manner.  It is expected that backends will give an
error when given an operation which cannot be implemented.  (The LLVM code
generator is not very helpful here at the moment, but hopefully that will
change.)

The implementation of atomics on LL/SC architectures (like ARM) is currently a
bit of a mess; there is a lot of copy-pasted code across targets, and the
representation is relatively unsuited to optimization (it would be nice to be
able to optimize loops involving cmpxchg etc.).

On x86, all atomic loads generate a ``MOV``. SequentiallyConsistent stores
generate an ``XCHG``, other stores generate a ``MOV``. SequentiallyConsistent
fences generate an ``MFENCE``, other fences do not cause any code to be
generated.  cmpxchg uses the ``LOCK CMPXCHG`` instruction.  ``atomicrmw xchg``
uses ``XCHG``, ``atomicrmw add`` and ``atomicrmw sub`` use ``XADD``, and all
other ``atomicrmw`` operations generate a loop with ``LOCK CMPXCHG``.  Depending
on the users of the result, some ``atomicrmw`` operations can be translated into
operations like ``LOCK AND``, but that does not work in general.

On ARM, MIPS, and many other RISC architectures, Acquire, Release, and
SequentiallyConsistent semantics require barrier instructions for every such
operation. Loads and stores generate normal instructions.  ``cmpxchg`` and
``atomicrmw`` can be represented using a loop with LL/SC-style instructions
which take some sort of exclusive lock on a cache line (``LDREX`` and ``STREX``
on ARM, etc.). At the moment, the IR does not provide any way to represent a
weak ``cmpxchg`` which would not require a loop.
