.. _amdgpu-async-operations:

===============================
 AMDGPU Asynchronous Operations
===============================

.. contents::
   :local:

Introduction
============

Asynchronous operations are operations that are completed independently at an
unspecified scope. A thread that requests one or more async operations can use
*asyncmarks* to track their completion.

Operations
==========

Async Instructions
------------------

The following instructions request async operations that transfer data between
global memory and LDS memory.

.. note::

   These listings are *merely representative*. The actual function signatures
   and supported architectures are documented in the :ref:`amdgpu-usage-guide`.

**GFX9 Async Instructions (LDS DMA)**

.. code-block:: llvm

  void @llvm.amdgcn.load.async.to.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.global.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.raw.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.struct.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr %src, ptr %dst)

**GFX12 Async Instructions**

.. code-block:: llvm

  void @llvm.amdgcn.global.load.async.to.lds.type(ptr %dst, ptr %src)
  void @llvm.amdgcn.global.store.async.from.lds.type(ptr %dst, ptr %src)
  void @llvm.amdgcn.cluster.load.async.to.lds.type(ptr %dst, ptr %src)

**GFX1250 Tensor DMA Instructions**

.. code-block:: llvm

  void @llvm.amdgcn.tensor.load.to.lds(...)
  void @llvm.amdgcn.tensor.store.from.lds(...)

Asyncmarks
----------

An *asyncmark* created by a thread can be used to track async operations
initiated by that thread. The abstract machine maintains a sequence of
asyncmarks during the execution of a function body, which excludes any
asyncmarks produced by calls to other functions encountered in the currently
executing function. The state of this sequence at each program point in the
function is called the *current sequence*.

``@llvm.amdgcn.asyncmark()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Appends an asyncmark to the current sequence.

``@llvm.amdgcn.wait.asyncmark(i16 %N)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensures that the length of the current sequence is at most ``N`` by removing
asyncmarks from the start of the sequence if it is more than ``N``.

Memory Consistency Model
========================

An ``asyncmark()`` operation ``X`` that inserts an asyncmark ``M`` is
*completed-at* a ``wait.asyncmark()`` operation ``Y`` in the same function body
if:

- ``X`` is *program-ordered* before ``Y``, and
- ``M`` is not in the current sequence at any operation ``Z`` that immediately
  follows ``Y`` in *program-order*.

When a thread executes an async *instruction* ``I``, it initiates a
corresponding async *operation* ``A``, and ``I`` is said to *happen-before*
``A``.

An async operation ``A`` initiated by an instruction ``I`` *happens-before* a
``wait.asyncmark()`` operation ``Y`` if there exists an ``asyncmark()``
operation ``X`` such that:

- ``I`` is *program-ordered* before ``X``, and
- ``X`` is *completed-at* ``Y``.

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

       // wait for the second block
       wait.asyncmark(1);

       // wait for the third block, including bar()
       wait.asyncmark(0);
   }

Implementation notes
====================

[This section is informational.]

Function Calls
--------------

In general, at a function call, if the caller uses sufficient waits to track
its own async operations, the actions performed by the callee cannot affect
correctness. But inlining such a call may result in redundant waits.

.. code-block:: c++

   void foo() {
     ...
     asyncmark();       // X
     ...                // no wait.asyncmark()
   }

   void bar() {
     asyncmark();       // B
     asyncmark();       // C
     foo();
     wait.asyncmark(1); // D
   }

Before inlining, it is unspecified whether ``X`` is *completed-at* ``D``, while
``C`` is **not** *completed-at* ``D``. The programmer can only rely on ``B``
being *completed-at* ``D``.

.. code-block:: c++

   void bar() {
     asyncmark();       // B
     asyncmark();       // C
     ...
     asyncmark();       // X
     ...                // no wait.asyncmark()
     wait.asyncmark(1); // D
   }

After inlining, ``C`` is also *completed-at* ``D`` and ``X`` is **not**
*completed-at* ``D``.

Conversely, a ``wait.asyncmark`` call inside a callee cannot be used to track
asyncmarks inserted by the caller, since this ``wait.asyncmark`` can only
observe the current sequence of the callee.

.. code-block:: c++

   void foo() {
     ...                // no asyncmark()
     wait.asyncmark(0); // Y
     ...
   }

   void bar() {
     asyncmark();       // B
     asyncmark();       // C
     foo();
     wait.asyncmark(1); // D
   }

In the above example, it is unspecified whether ``B`` and ``C`` in ``bar()`` are
*completed-at* ``Y``, because they are not included in the sequence that can be
examined at ``Y``.

.. code-block:: c++

   void bar() {
     asyncmark();       // B
     asyncmark();       // C
     ...                // no asyncmark()
     wait.asyncmark(0); // Y
     ...
     wait.asyncmark(1); // D
   }

After inlining, both ``B`` and ``C`` are *completed-at* ``Y``.

Optimization
------------

The implementation may eliminate asyncmark/wait intrinsics in the following
cases. These are just examples and not meant to be an exhaustive list.

1. An ``asyncmark`` operation which remains in the current sequence along every
   path that reaches the function exit.

   .. code-block:: c++

      void foo() {
        ...
        asyncmark();       // X
        ...                // no wait.asyncmark()
      }

   Here, ``X`` can be eliminated.

2. A ``wait.asyncmark`` which sees an empty sequence of asyncmarks along every
   path that reaches it.

   .. code-block:: c++

      void foo() {
        ...                // no asyncmark()
        wait.asyncmark(0); // Y
        ...
      }

    Here, ``Y`` can be eliminated.
