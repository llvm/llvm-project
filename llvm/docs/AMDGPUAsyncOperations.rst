.. _amdgpu-async-operations:

===============================
 AMDGPU Asynchronous Operations
===============================

.. contents::
   :local:

Introduction
============

Asynchronous operations are memory transfers (usually between the global memory
and LDS) that are completed independently at an unspecified scope. A thread that
requests one or more asynchronous transfers can use *async marks* to track
their completion. The thread waits for each mark to be *completed*, which
indicates that requests initiated in program order before this mark have also
completed.

Operations
==========

Memory Accesses
---------------

LDS DMA Operations
^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  ; "Legacy" LDS DMA operations
  void @llvm.amdgcn.load.async.to.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.global.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.raw.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.struct.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr %src, ptr %dst)

Request an async operation that copies the specified number of bytes from the
global/buffer pointer ``%src`` to the LDS pointer ``%dst``.

.. note::

   The above listing is *merely representative*. The actual function signatures
   are identical to their non-async variants, and supported only on the
   corresponding architectures (GFX9 and GFX10).

Async Mark Operations
---------------------

An *async mark* in the abstract machine tracks all the async operations that
are program ordered before that mark. A mark M is said to be *completed*
only when all async operations program ordered before M are reported by the
implementation as having finished, and it is said to be *outstanding* otherwise.

Thus we have the following sufficient condition:

  An async operation X is *completed* at a program point P if there exists a
  mark M such that X is program ordered before M, M is program ordered before
  P, and M is completed. X is said to be *outstanding* at P otherwise.

The abstract machine maintains a sequence of *async marks* during the
execution of a function body, which excludes any marks produced by calls to
other functions encountered in the currently executing function.


``@llvm.amdgcn.asyncmark()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When executed, inserts an async mark in the sequence associated with the
currently executing function body.

``@llvm.amdgcn.wait.asyncmark(i16 %N)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Waits until there are at most N outstanding marks in the sequence associated
with the currently executing function body.

Memory Consistency Model
========================

Each asynchronous operation consists of a non-atomic read on the source and a
non-atomic write on the destination. Async "LDS DMA" intrinsics result in async
accesses that guarantee visibility relative to other memory operations as
follows:

  An asynchronous operation `A` program ordered before an overlapping memory
  operation `X` happens-before `X` only if `A` is completed before `X`.

  A memory operation `X` program ordered before an overlapping asynchronous
  operation `A` happens-before `A`.

.. note::

   The *only if* in the above wording implies that unlike the default LLVM
   memory model, certain program order edges are not automatically included in
   ``happens-before``.

Examples
========

Uneven blocks of async transfers
--------------------------------

.. code-block:: c++

   void foo(global int *g, local int *l) {
     // first block
     async_load_to_lds(l, g);
     async_load_to_lds(l, g);
     async_load_to_lds(l, g);
     asyncmark();

     // second block; longer
     async_load_to_lds(l, g);
     async_load_to_lds(l, g);
     async_load_to_lds(l, g);
     async_load_to_lds(l, g);
     async_load_to_lds(l, g);
     asyncmark();

     // third block; shorter
     async_load_to_lds(l, g);
     async_load_to_lds(l, g);
     asyncmark();

     // Wait for first block
     wait.asyncmark(2);
   }

Software pipeline
-----------------

.. code-block:: c++

   void foo(global int *g, local int *l) {
     // first block
     asyncmark();

     // second block
     asyncmark();

     // third block
     asyncmark();

     for (;;) {
       wait.asyncmark(2);
       // use data

       // next block
       asyncmark();
     }

     // flush one block
     wait.asyncmark(2);

     // flush one more block
     wait.asyncmark(1);

     // flush last block
     wait.asyncmark(0);
   }

Ordinary function call
----------------------

.. code-block:: c++

   extern void bar(); // may or may not make async calls

   void foo(global int *g, local int *l) {
       // first block
       asyncmark();

       // second block
       asyncmark();

       // function call
       bar();

       // third block
       asyncmark();

       wait.asyncmark(1); // will wait for at least the second block, possibly including bar()
       wait.asyncmark(0); // will wait for third block, including bar()
   }

Implementation notes
====================

[This section is informational.]

Optimization
------------

The implementation may eliminate async mark/wait intrinsics in the following cases:

1. An ``asyncmark`` operation which is not included in the wait count of a later
   wait operation in the current function. In particular, an ``asyncmark`` which
   is not post-dominated by any ``wait.asyncmark``.
2. A ``wait.asyncmark`` whose wait count is more than the outstanding async
   marks at that point. In particular, a ``wait.asyncmark`` that is not
   dominated by any ``asyncmark``.

In general, at a function call, if the caller uses sufficient waits to track
its own async operations, the actions performed by the callee cannot affect
correctness. But inlining such a call may result in redundant waits.

.. code-block:: c++

   void foo() {
     asyncmark(); // A
   }

   void bar() {
     asyncmark(); // B
     asyncmark(); // C
     foo();
     wait.asyncmark(1);
   }

Before inlining, the ``wait.asyncmark`` waits for mark B to be completed.

.. code-block:: c++

   void foo() {
   }

   void bar() {
     asyncmark(); // B
     asyncmark(); // C
     asyncmark(); // A from call to foo()
     wait.asyncmark(1);
   }

After inlining, the asyncmark-wait now waits for mark C to complete, which is
longer than necessary. Ideally, the optimizer should have eliminated mark A in
the body of foo() itself.
