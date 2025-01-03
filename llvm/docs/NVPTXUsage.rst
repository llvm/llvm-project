=============================
User Guide for NVPTX Back-end
=============================

.. contents::
   :local:
   :depth: 3


Introduction
============

To support GPU programming, the NVPTX back-end supports a subset of LLVM IR
along with a defined set of conventions used to represent GPU programming
concepts. This document provides an overview of the general usage of the back-
end, including a description of the conventions used and the set of accepted
LLVM IR.

.. note::

   This document assumes a basic familiarity with CUDA and the PTX
   assembly language. Information about the CUDA Driver API and the PTX assembly
   language can be found in the `CUDA documentation
   <http://docs.nvidia.com/cuda/index.html>`_.



Conventions
===========

Marking Functions as Kernels
----------------------------

In PTX, there are two types of functions: *device functions*, which are only
callable by device code, and *kernel functions*, which are callable by host
code. By default, the back-end will emit device functions. Metadata is used to
declare a function as a kernel function. This metadata is attached to the
``nvvm.annotations`` named metadata object, and has the following format:

.. code-block:: text

   !0 = !{<function-ref>, metadata !"kernel", i32 1}

The first parameter is a reference to the kernel function. The following
example shows a kernel function calling a device function in LLVM IR. The
function ``@my_kernel`` is callable from host code, but ``@my_fmad`` is not.

.. code-block:: llvm

    define float @my_fmad(float %x, float %y, float %z) {
      %mul = fmul float %x, %y
      %add = fadd float %mul, %z
      ret float %add
    }

    define void @my_kernel(ptr %ptr) {
      %val = load float, ptr %ptr
      %ret = call float @my_fmad(float %val, float %val, float %val)
      store float %ret, ptr %ptr
      ret void
    }

    !nvvm.annotations = !{!1}
    !1 = !{ptr @my_kernel, !"kernel", i32 1}

When compiled, the PTX kernel functions are callable by host-side code.


.. _address_spaces:

Address Spaces
--------------

The NVPTX back-end uses the following address space mapping:

   ============= ======================
   Address Space Memory Space
   ============= ======================
   0             Generic
   1             Global
   2             Internal Use
   3             Shared
   4             Constant
   5             Local
   ============= ======================

Every global variable and pointer type is assigned to one of these address
spaces, with 0 being the default address space. Intrinsics are provided which
can be used to convert pointers between the generic and non-generic address
spaces.

As an example, the following IR will define an array ``@g`` that resides in
global device memory.

.. code-block:: llvm

    @g = internal addrspace(1) global [4 x i32] [ i32 0, i32 1, i32 2, i32 3 ]

LLVM IR functions can read and write to this array, and host-side code can
copy data to it by name with the CUDA Driver API.

Note that since address space 0 is the generic space, it is illegal to have
global variables in address space 0.  Address space 0 is the default address
space in LLVM, so the ``addrspace(N)`` annotation is *required* for global
variables.


Triples
-------

The NVPTX target uses the module triple to select between 32/64-bit code
generation and the driver-compiler interface to use. The triple architecture
can be one of ``nvptx`` (32-bit PTX) or ``nvptx64`` (64-bit PTX). The
operating system should be one of ``cuda`` or ``nvcl``, which determines the
interface used by the generated code to communicate with the driver.  Most
users will want to use ``cuda`` as the operating system, which makes the
generated PTX compatible with the CUDA Driver API.

Example: 32-bit PTX for CUDA Driver API: ``nvptx-nvidia-cuda``

Example: 64-bit PTX for CUDA Driver API: ``nvptx64-nvidia-cuda``



.. _nvptx_intrinsics:

NVPTX Intrinsics
================

Reading PTX Special Registers
-----------------------------

'``llvm.nvvm.read.ptx.sreg.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
    declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
    declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()

Overview:
"""""""""

The '``@llvm.nvvm.read.ptx.sreg.*``' intrinsics provide access to the PTX
special registers, in particular the kernel launch bounds.  These registers
map in the following way to CUDA builtins:

   ============ =====================================
   CUDA Builtin PTX Special Register Intrinsic
   ============ =====================================
   ``threadId`` ``@llvm.nvvm.read.ptx.sreg.tid.*``
   ``blockIdx`` ``@llvm.nvvm.read.ptx.sreg.ctaid.*``
   ``blockDim`` ``@llvm.nvvm.read.ptx.sreg.ntid.*``
   ``gridDim``  ``@llvm.nvvm.read.ptx.sreg.nctaid.*``
   ============ =====================================


Barriers
--------

'``llvm.nvvm.barrier0``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.barrier0()

Overview:
"""""""""

The '``@llvm.nvvm.barrier0()``' intrinsic emits a PTX ``bar.sync 0``
instruction, equivalent to the ``__syncthreads()`` call in CUDA.

Electing a thread
-----------------

'``llvm.nvvm.elect.sync``'
^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare {i32, i1} @llvm.nvvm.elect.sync(i32 %membermask)

Overview:
"""""""""

The '``@llvm.nvvm.elect.sync``' intrinsic generates the ``elect.sync``
PTX instruction, which elects one predicated active leader thread from
a set of threads specified by ``membermask``. The behavior is undefined
if the executing thread is not in ``membermask``. The laneid of the
elected thread is captured in the i32 return value. The i1 return
value is set to ``True`` for the leader thread and ``False`` for all
the other threads. Election of a leader thread happens deterministically,
i.e. the same leader thread is elected for the same ``membermask``
every time. For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync>`_.

Membar/Fences
-------------

'``llvm.nvvm.fence.proxy.tensormap_generic.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.fence.proxy.tensormap_generic.release.cta()
  declare void @llvm.nvvm.fence.proxy.tensormap_generic.release.cluster()
  declare void @llvm.nvvm.fence.proxy.tensormap_generic.release.gpu()
  declare void @llvm.nvvm.fence.proxy.tensormap_generic.release.sys()

  declare void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr %addr, i32 %size)
  declare void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cluster(ptr %addr, i32 %size)
  declare void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu(ptr %addr, i32 %size)
  declare void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys(ptr %addr, i32 %size)

Overview:
"""""""""

The ``@llvm.nvvm.fence.proxy.tensormap_generic.*`` is a uni-directional fence used to establish ordering between a prior memory access performed via the generic `proxy<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies>_` and a subsequent memory access performed via the tensormap proxy. ``nvvm.fence.proxy.tensormap_generic.release`` can form a release sequence that synchronizes with an acquire sequence that contains the ``nvvm.fence.proxy.tensormap_generic.acquire`` proxy fence. The following table describes the mapping between LLVM Intrinsic and the PTX instruction:

  ====================================================== =========================================================
  NVVM Intrinsic                                         PTX Instruction
  ====================================================== =========================================================
  ``@llvm.nvvm.fence.proxy.tensormap_generic.release.*`` ``fence.proxy.tensormap::generic.release.*``
  ``@llvm.nvvm.fence.proxy.tensormap_generic.acquire.*`` ``fence.proxy.tensormap::generic.acquire.* [addr], size``
  ====================================================== =========================================================

The address operand ``addr`` and the operand ``size`` together specify the memory range ``[addr, addr+size)`` on which the ordering guarantees on the memory accesses across the proxies is to be provided. The only supported value for the ``size`` operand is ``128`` and must be an immediate. Generic Addressing is used unconditionally, and the address specified by the operand addr must fall within the ``.global`` state space. Otherwise, the behavior is undefined. For more information, see `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`_.

Address Space Intrinsics
------------------------

'``llvm.nvvm.isspacep.*``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i1 @llvm.nvvm.isspacep.const(ptr %p)
    declare i1 @llvm.nvvm.isspacep.global(ptr %p)
    declare i1 @llvm.nvvm.isspacep.local(ptr %p)
    declare i1 @llvm.nvvm.isspacep.shared(ptr %p)
    declare i1 @llvm.nvvm.isspacep.shared.cluster(ptr %p)

Overview:
"""""""""

The '``llvm.nvvm.isspacep.*``' intrinsics determine whether the provided generic
pointer references memory which falls within a particular address space.

Semantics:
""""""""""

If the given pointer in the generic address space refers to memory which falls
within the state space of the intrinsic (and therefore could be safely address
space casted to this space), 1 is returned, otherwise 0 is returned.

Arithmetic Intrinsics
---------------------

'``llvm.nvvm.idp2a.[us].[us]``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.idp2a.s.s(i32 %a, i32 %b, i1 immarg %is.hi, i32 %c)
    declare i32 @llvm.nvvm.idp2a.s.u(i32 %a, i32 %b, i1 immarg %is.hi, i32 %c)
    declare i32 @llvm.nvvm.idp2a.u.s(i32 %a, i32 %b, i1 immarg %is.hi, i32 %c)
    declare i32 @llvm.nvvm.idp2a.u.u(i32 %a, i32 %b, i1 immarg %is.hi, i32 %c)


Overview:
"""""""""

The '``llvm.nvvm.idp2a.[us].[us]``' intrinsics performs a 2-element vector dot
product followed by addition. They corresponds directly to the ``dp2a`` PTX 
instruction.

Semantics:
""""""""""

The 32-bit value in ``%a`` is broken into 2 16-bit values which are extended to
32 bits. For the '``llvm.nvvm.idp2a.u.[us]``' variants zero-extension is used,
while for the '``llvm.nvvm.idp2a.s.[us]``' sign-extension is used. Two bytes are
selected from ``%b``, if ``%is.hi`` is true, the most significant bytes are
selected, otherwise the least significant bytes are selected. These bytes are
then extended to 32-bits. For the '``llvm.nvvm.idp2a.[us].u``' variants
zero-extension is used, while for the '``llvm.nvvm.idp2a.[us].s``'
sign-extension is used. The dot product of these 2-element vectors is added to
``%c`` to produce the return.


'``llvm.nvvm.idp4a.[us].[us]``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.idp4a.s.s(i32 %a, i32 %b, i32 %c)
    declare i32 @llvm.nvvm.idp4a.s.u(i32 %a, i32 %b, i32 %c)
    declare i32 @llvm.nvvm.idp4a.u.s(i32 %a, i32 %b, i32 %c)
    declare i32 @llvm.nvvm.idp4a.u.u(i32 %a, i32 %b, i32 %c)

Overview:
"""""""""

The '``llvm.nvvm.idp4a.[us].[us]``' intrinsics perform a 4-element vector dot
product followed by addition. They corresponds directly to the ``dp4a`` PTX
instruction.

Semantics:
""""""""""

Each of the 4 bytes in both ``%a`` and ``%b`` are extended to 32-bit integers
forming 2 ``<4 x i32>``. For ``%a``, zero-extension is used in the
'``llvm.nvvm.idp4a.u.[us]``' variants, while sign-extension is used with
'``llvm.nvvm.idp4a.s.[us]``' variants. Similarly, for ``%b``, zero-extension is
used in the '``llvm.nvvm.idp4a.[us].u``' variants, while sign-extension is used
with '``llvm.nvvm.idp4a.[us].s``' variants. The dot product of these 4-element
vectors is added to ``%c`` to produce the return.

Bit Manipulation Intrinsics
---------------------------

'``llvm.nvvm.fshl.clamp.*``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.fshl.clamp.i32(i32 %hi, i32 %lo, i32 %n)

Overview:
"""""""""

The '``llvm.nvvm.fshl.clamp``' family of intrinsics performs a clamped funnel
shift left. These intrinsics are very similar to '``llvm.fshl``', except the
shift ammont is clamped at the integer width (instead of modulo it). Currently,
only ``i32`` is supported.

Semantics:
""""""""""

The '``llvm.nvvm.fshl.clamp``' family of intrinsic functions performs a clamped
funnel shift left: the first two values are concatenated as { %hi : %lo } (%hi
is the most significant bits of the wide value), the combined value is shifted
left, and the most significant bits are extracted to produce a result that is
the same size as the original arguments. The shift amount is the minimum of the
value of %n and the bit width of the integer type.

'``llvm.nvvm.fshr.clamp.*``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.fshr.clamp.i32(i32 %hi, i32 %lo, i32 %n)

Overview:
"""""""""

The '``llvm.nvvm.fshr.clamp``' family of intrinsics perform a clamped funnel
shift right. These intrinsics are very similar to '``llvm.fshr``', except the
shift ammont is clamped at the integer width (instead of modulo it). Currently,
only ``i32`` is supported.

Semantics:
""""""""""

The '``llvm.nvvm.fshr.clamp``' family of intrinsic functions performs a clamped
funnel shift right: the first two values are concatenated as { %hi : %lo } (%hi
is the most significant bits of the wide value), the combined value is shifted
right, and the least significant bits are extracted to produce a result that is
the same size as the original arguments. The shift amount is the minimum of the
value of %n and the bit width of the integer type.

'``llvm.nvvm.flo.u.*``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.flo.u.i32(i32 %a, i1 %shiftamt)
    declare i32 @llvm.nvvm.flo.u.i64(i64 %a, i1 %shiftamt)

Overview:
"""""""""

The '``llvm.nvvm.flo.u``' family of intrinsics identifies the bit position of the
leading one, returning either it's offset from the most or least significant bit.

Semantics:
""""""""""

The '``llvm.nvvm.flo.u``' family of intrinsics returns the bit position of the
most significant 1. If %shiftamt is true, The result is the shift amount needed
to left-shift the found bit into the most-significant bit position, otherwise
the result is the shift amount needed to right-shift the found bit into the
least-significant bit position. 0xffffffff is returned if no 1 bit is found.

'``llvm.nvvm.flo.s.*``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.flo.s.i32(i32 %a, i1 %shiftamt)
    declare i32 @llvm.nvvm.flo.s.i64(i64 %a, i1 %shiftamt)

Overview:
"""""""""

The '``llvm.nvvm.flo.s``' family of intrinsics identifies the bit position of the
leading non-sign bit, returning either it's offset from the most or least
significant bit.

Semantics:
""""""""""

The '``llvm.nvvm.flo.s``' family of intrinsics returns the bit position of the
most significant 0 for negative inputs and the most significant 1 for 
non-negative inputs. If %shiftamt is true, The result is the shift amount needed
to left-shift the found bit into the most-significant bit position, otherwise
the result is the shift amount needed to right-shift the found bit into the
least-significant bit position. 0xffffffff is returned if no 1 bit is found.

TMA family of Intrinsics
------------------------

'``llvm.nvvm.cp.async.bulk.tensor.g2s.tile.[1-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(3) %dst, ptr addrspace(3) %bar, ptr %tensor_map, i32 %d0, i16 %mc, i64 %ch, i1 %flag_mc, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(..., i32 %d0, i32 %d1, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(..., i32 %d0, i32 %d1, i32 %d2, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.g2s.tile.[1-5]d``' intrinsics
correspond to the ``cp.async.bulk.tensor.[1-5]d.*`` set of PTX instructions.
These instructions initiate an asynchronous copy of tensor data from
global memory to shared::cluster memory (indicated by the ``g2s`` prefix)
in ``tile`` mode. In tile mode, the multi-dimensional layout of the
source tensor is preserved at the destination. The dimension of the
tensor data ranges from 1d to 5d with the coordinates specified
by the ``i32 %d0 ... i32 %d4`` arguments.

* The last two arguments to these intrinsics are boolean flags
  indicating support for cache_hint and/or multicast modifiers.
  These flag arguments must be compile-time constants. The backend
  looks through these flags and lowers the intrinsics appropriately.

* The Nth argument (denoted by ``i1 flag_ch``) when set, indicates
  a valid cache_hint (``i64 %ch``) and generates the ``.L2::cache_hint``
  variant of the PTX instruction.

* The [N-1]th argument (denoted by ``i1 flag_mc``) when set, indicates
  the presence of a multicast mask (``i16 %mc``) and generates the PTX
  instruction with the ``.multicast::cluster`` modifier.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`_.

'``llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.[3-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(3) %dst, ptr addrspace(3) %bar, ptr %tensor_map, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch, i1 %flag_mc, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.[3-5]d``' intrinsics
correspond to the ``cp.async.bulk.tensor.[1-5]d.*`` set of PTX instructions.
These instructions initiate an asynchronous copy of tensor data from
global memory to shared::cluster memory (indicated by the ``g2s`` prefix)
in ``im2col`` mode. In im2col mode, some dimensions of the source tensor
are unrolled into a single dimensional column at the destination. In this
mode, the tensor has to be at least three-dimensional. Along with the tensor
coordinates, im2col offsets are also specified (denoted by
``i16 im2col0...i16 %im2col2``). The number of im2col offsets is two less
than the number of dimensions of the tensor operation. The last two arguments
to these intrinsics are boolean flags, with the same functionality as described
in the ``tile`` mode intrinsics above.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`_.

'``llvm.nvvm.cp.async.bulk.tensor.s2g.tile.[1-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d(..., i32 %d0, i32 %d1, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.3d(..., i32 %d0, i32 %d1, i32 %d2, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.s2g.tile.[1-5]d``' intrinsics
correspond to the ``cp.async.bulk.tensor.[1-5]d.*`` set of PTX instructions.
These instructions initiate an asynchronous copy of tensor data from
shared::cta to global memory (indicated by the ``s2g`` prefix)
in ``tile`` mode. The dimension of the tensor data ranges from 1d to 5d
with the coordinates specified by the ``i32 %d0 ... i32 %d4`` arguments.

* The last argument to these intrinsics is a boolean flag
  indicating support for cache_hint. This flag argument must
  be a compile-time constant. When set, it indicates a valid
  cache_hint (``i64 %ch``) and generates the ``.L2::cache_hint``
  variant of the PTX instruction.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`_.

'``llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.[3-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.3d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i32 %d1, i32 %d2, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.[1-5]d``' intrinsics
correspond to the ``cp.async.bulk.tensor.[1-5]d.*`` set of PTX instructions.
These instructions initiate an asynchronous copy of tensor data from
shared::cta to global memory (indicated by the ``s2g`` prefix)
in ``im2col`` mode. In this mode, the tensor has to be at least
three-dimensional. Unlike the ``g2s`` variants, there are no
im2col_offsets for these intrinsics. The last argument to these
intrinsics is a boolean flag, with the same functionality as
described in the ``s2g.tile`` mode intrinsics above.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`_.

'``llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.[1-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(..., i32 %d0, i32 %d1, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(..., i32 %d0, i32 %d1, i32 %d2, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.[1-5]d``' intrinsics
correspond to the ``cp.async.bulk.prefetch.tensor.[1-5]d.L2.global*`` set
of PTX instructions. These instructions initiate an asynchronous prefetch
of tensor data from global memory to the L2 cache. In tile mode, the
multi-dimensional layout of the source tensor is preserved at the destination.
The dimension of the tensor data ranges from 1d to 5d with the coordinates
specified by the ``i32 %d0 ... i32 %d4`` arguments.

* The last argument to these intrinsics is a boolean flag
  indicating support for cache_hint. This flag argument must
  be a compile-time constant. When set, it indicates a valid
  cache_hint (``i64 %ch``) and generates the ``.L2::cache_hint``
  variant of the PTX instruction.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>`_.

'``llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.[3-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %tensor_map, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.[3-5]d``' intrinsics
correspond to the ``cp.async.bulk.prefetch.tensor.[1-5]d.L2.global*`` set
of PTX instructions. These instructions initiate an asynchronous prefetch
of tensor data from global memory to the L2 cache. In im2col mode, some
dimensions of the source tensor are unrolled into a single dimensional
column at the destination. In this mode, the tensor has to be at least
three-dimensional. Along with the tensor coordinates, im2col offsets are
also specified (denoted by ``i16 im2col0...i16 %im2col2``). The number
of im2col offsets is two less than the number of dimensions of the tensor
operation. The last argument to these intrinsics is a boolean flag, with
the same functionality as described in the ``tile`` mode intrinsics above.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>`_.

'``llvm.nvvm.cp.async.bulk.tensor.reduce.[red_op].tile.[1-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.1d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i64 %ch, i1 %flag_ch)

  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.tile.2d(..., i32 %d0, i32 %d1, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.tile.3d(..., i32 %d0, i32 %d1, i32 %d2, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.tile.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.tile.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.tile.[1-5]d``' intrinsics
correspond to the ``cp.reduce.async.bulk.tensor.[1-5]d.*`` set of PTX instructions.
These instructions initiate an asynchronous reduction operation of tensor data
in global memory with the tensor data in shared{::cta} memory, using ``tile`` mode.
The dimension of the tensor data ranges from 1d to 5d with the coordinates
specified by the ``i32 %d0 ... i32 %d4`` arguments. The supported reduction
operations are {add, min, max, inc, dec, and, or, xor} as described in the
``tile.1d`` intrinsics.

* The last argument to these intrinsics is a boolean flag
  indicating support for cache_hint. This flag argument must
  be a compile-time constant. When set, it indicates a valid
  cache_hint (``i64 %ch``) and generates the ``.L2::cache_hint``
  variant of the PTX instruction.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor>`_.

'``llvm.nvvm.cp.async.bulk.tensor.reduce.[red_op].im2col.[3-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.im2col.3d(ptr addrspace(3) %src, ptr %tensor_map, i32 %d0, i32 %d1, i32 %d2, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.im2col.4d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, ...)
  declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.im2col.5d(..., i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, ...)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.tensor.reduce.<red_op>.im2col.[3-5]d``' intrinsics
correspond to the ``cp.reduce.async.bulk.tensor.[3-5]d.*`` set of PTX instructions.
These instructions initiate an asynchronous reduction operation of tensor data
in global memory with the tensor data in shared{::cta} memory, using ``im2col`` mode.
In this mode, the tensor has to be at least three-dimensional. The supported reduction
operations supported are the same as the ones in the tile mode. The last argument to
these intrinsics is a boolean flag, with the same functionality as described in the
``tile`` mode intrinsics above.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor>`_.

Warp Group Intrinsics
---------------------

'``llvm.nvvm.wgmma.fence.sync.aligned``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.wgmma.fence.sync.aligned()

Overview:
"""""""""

The '``@llvm.nvvm.wgmma.fence.sync.aligned``' intrinsic generates the
``wgmma.fence.sync.aligned`` PTX instruction, which establishes an ordering
between prior accesses to any warpgroup registers and subsequent accesses to
the same registers by a ``wgmma.mma_async`` instruction.

The ``wgmma.fence`` instruction must be issued by all warps of the warpgroup in
the following locations:

* Before the first ``wgmma.mma_async`` operation in a warpgroup.
* Between a register access by a thread in the warpgroup and any
  ``wgmma.mma_async`` instruction that accesses the same registers, except when
  these are accumulator register accesses across multiple ``wgmma.mma_async``
  instructions of the same shape in which case an ordering guarantee is
  provided by default.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence>`_.

'``llvm.nvvm.wgmma.commit_group.sync.aligned``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.wgmma.commit_group.sync.aligned()

Overview:
"""""""""

The '``@llvm.nvvm.wgmma.commit_group.sync.aligned``' intrinsic generates the
``wgmma.commit_group.sync.aligned`` PTX instruction, which creates a new
wgmma-group per warpgroup and batches all prior ``wgmma.mma_async``
instructions initiated by the executing warp but not committed to any
wgmma-group into the new wgmma-group. If there are no uncommitted ``wgmma
mma_async`` instructions then, ``wgmma.commit_group`` results in an empty
wgmma-group.

An executing thread can wait for the completion of all ``wgmma.mma_async``
operations in a wgmma-group by using ``wgmma.wait_group``.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group>`_.

'``llvm.nvvm.wgmma.wait_group.sync.aligned``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.wgmma.wait_group.sync.aligned(i64 immarg N)

Overview:
"""""""""

The '``@llvm.nvvm.wgmma.wait_group.sync.aligned``' intrinsic generates the
``wgmma.commit_group.sync.aligned N`` PTX instruction, which will cause the
executing thread to wait until only ``N`` or fewer of the most recent
wgmma-groups are pending and all the prior wgmma-groups committed by the
executing threads are complete. For example, when ``N`` is 0, the executing
thread waits on all the prior wgmma-groups to complete. Operand ``N`` is an
integer constant.

Accessing the accumulator register or the input register containing the
fragments of matrix A of a ``wgmma.mma_async`` instruction without first
performing a ``wgmma.wait_group`` instruction that waits on a wgmma-group
including that ``wgmma.mma_async`` instruction is undefined behavior.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group>`_.

Other Intrinsics
----------------

For the full set of NVPTX intrinsics, please see the
``include/llvm/IR/IntrinsicsNVVM.td`` file in the LLVM source tree.


.. _libdevice:

Linking with Libdevice
======================

The CUDA Toolkit comes with an LLVM bitcode library called ``libdevice`` that
implements many common mathematical functions. This library can be used as a
high-performance math library for any compilers using the LLVM NVPTX target.
The library can be found under ``nvvm/libdevice/`` in the CUDA Toolkit and
there is a separate version for each compute architecture.

For a list of all math functions implemented in libdevice, see
`libdevice Users Guide <http://docs.nvidia.com/cuda/libdevice-users-guide/index.html>`_.

To accommodate various math-related compiler flags that can affect code
generation of libdevice code, the library code depends on a special LLVM IR
pass (``NVVMReflect``) to handle conditional compilation within LLVM IR. This
pass looks for calls to the ``@__nvvm_reflect`` function and replaces them
with constants based on the defined reflection parameters. Such conditional
code often follows a pattern:

.. code-block:: c++

  float my_function(float a) {
    if (__nvvm_reflect("FASTMATH"))
      return my_function_fast(a);
    else
      return my_function_precise(a);
  }

The default value for all unspecified reflection parameters is zero.

The ``NVVMReflect`` pass should be executed early in the optimization
pipeline, immediately after the link stage. The ``internalize`` pass is also
recommended to remove unused math functions from the resulting PTX. For an
input IR module ``module.bc``, the following compilation flow is recommended:

The ``NVVMReflect`` pass will attempt to remove dead code even without
optimizations. This allows potentially incompatible instructions to be avoided
at all optimizations levels by using the ``__CUDA_ARCH`` argument.

1. Save list of external functions in ``module.bc``
2. Link ``module.bc`` with ``libdevice.compute_XX.YY.bc``
3. Internalize all functions not in list from (1)
4. Eliminate all unused internal functions
5. Run ``NVVMReflect`` pass
6. Run standard optimization pipeline

.. note::

  ``linkonce`` and ``linkonce_odr`` linkage types are not suitable for the
  libdevice functions. It is possible to link two IR modules that have been
  linked against libdevice using different reflection variables.

Since the ``NVVMReflect`` pass replaces conditionals with constants, it will
often leave behind dead code of the form:

.. code-block:: llvm

  entry:
    ..
    br i1 true, label %foo, label %bar
  foo:
    ..
  bar:
    ; Dead code
    ..

Therefore, it is recommended that ``NVVMReflect`` is executed early in the
optimization pipeline before dead-code elimination.

The NVPTX TargetMachine knows how to schedule ``NVVMReflect`` at the beginning
of your pass manager; just use the following code when setting up your pass
manager and the PassBuilder will use ``registerPassBuilderCallbacks`` to let
NVPTXTargetMachine::registerPassBuilderCallbacks add the pass to the
pass manager:

.. code-block:: c++

    std::unique_ptr<TargetMachine> TM = ...;
    PassBuilder PB(TM);
    ModulePassManager MPM;
    PB.parsePassPipeline(MPM, ...);

Reflection Parameters
---------------------

The libdevice library currently uses the following reflection parameters to
control code generation:

==================== ======================================================
Flag                 Description
==================== ======================================================
``__CUDA_FTZ=[0,1]`` Use optimized code paths that flush subnormals to zero
==================== ======================================================

The value of this flag is determined by the "nvvm-reflect-ftz" module flag.
The following sets the ftz flag to 1.

.. code-block:: llvm

    !llvm.module.flags = !{!0}
    !0 = !{i32 4, !"nvvm-reflect-ftz", i32 1}

(``i32 4`` indicates that the value set here overrides the value in another
module we link with.  See the `LangRef <LangRef.html#module-flags-metadata>`
for details.)

Executing PTX
=============

The most common way to execute PTX assembly on a GPU device is to use the CUDA
Driver API. This API is a low-level interface to the GPU driver and allows for
JIT compilation of PTX code to native GPU machine code.

Initializing the Driver API:

.. code-block:: c++

    CUdevice device;
    CUcontext context;

    // Initialize the driver API
    cuInit(0);
    // Get a handle to the first compute device
    cuDeviceGet(&device, 0);
    // Create a compute device context
    cuCtxCreate(&context, 0, device);

JIT compiling a PTX string to a device binary:

.. code-block:: c++

    CUmodule module;
    CUfunction function;

    // JIT compile a null-terminated PTX string
    cuModuleLoadData(&module, (void*)PTXString);

    // Get a handle to the "myfunction" kernel function
    cuModuleGetFunction(&function, module, "myfunction");

For full examples of executing PTX assembly, please see the `CUDA Samples
<https://developer.nvidia.com/cuda-downloads>`_ distribution.


Common Issues
=============

ptxas complains of undefined function: __nvvm_reflect
-----------------------------------------------------

When linking with libdevice, the ``NVVMReflect`` pass must be used. See
:ref:`libdevice` for more information.


Tutorial: A Simple Compute Kernel
=================================

To start, let us take a look at a simple compute kernel written directly in
LLVM IR. The kernel implements vector addition, where each thread computes one
element of the output vector C from the input vectors A and B.  To make this
easier, we also assume that only a single CTA (thread block) will be launched,
and that it will be one dimensional.


The Kernel
----------

.. code-block:: llvm

  target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
  target triple = "nvptx64-nvidia-cuda"

  ; Intrinsic to read X component of thread ID
  declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind

  define void @kernel(ptr addrspace(1) %A,
                      ptr addrspace(1) %B,
                      ptr addrspace(1) %C) {
  entry:
    ; What is my ID?
    %id = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind

    ; Compute pointers into A, B, and C
    %ptrA = getelementptr float, ptr addrspace(1) %A, i32 %id
    %ptrB = getelementptr float, ptr addrspace(1) %B, i32 %id
    %ptrC = getelementptr float, ptr addrspace(1) %C, i32 %id

    ; Read A, B
    %valA = load float, ptr addrspace(1) %ptrA, align 4
    %valB = load float, ptr addrspace(1) %ptrB, align 4

    ; Compute C = A + B
    %valC = fadd float %valA, %valB

    ; Store back to C
    store float %valC, ptr addrspace(1) %ptrC, align 4

    ret void
  }

  !nvvm.annotations = !{!0}
  !0 = !{ptr @kernel, !"kernel", i32 1}


We can use the LLVM ``llc`` tool to directly run the NVPTX code generator:

.. code-block:: text

  # llc -mcpu=sm_20 kernel.ll -o kernel.ptx


.. note::

  If you want to generate 32-bit code, change ``p:64:64:64`` to ``p:32:32:32``
  in the module data layout string and use ``nvptx-nvidia-cuda`` as the
  target triple.


The output we get from ``llc`` (as of LLVM 3.4):

.. code-block:: text

  //
  // Generated by LLVM NVPTX Back-End
  //

  .version 3.1
  .target sm_20
  .address_size 64

    // .globl kernel
                                          // @kernel
  .visible .entry kernel(
    .param .u64 kernel_param_0,
    .param .u64 kernel_param_1,
    .param .u64 kernel_param_2
  )
  {
    .reg .f32   %f<4>;
    .reg .s32   %r<2>;
    .reg .s64   %rl<8>;

  // %bb.0:                                // %entry
    ld.param.u64    %rl1, [kernel_param_0];
    mov.u32         %r1, %tid.x;
    mul.wide.s32    %rl2, %r1, 4;
    add.s64         %rl3, %rl1, %rl2;
    ld.param.u64    %rl4, [kernel_param_1];
    add.s64         %rl5, %rl4, %rl2;
    ld.param.u64    %rl6, [kernel_param_2];
    add.s64         %rl7, %rl6, %rl2;
    ld.global.f32   %f1, [%rl3];
    ld.global.f32   %f2, [%rl5];
    add.f32         %f3, %f1, %f2;
    st.global.f32   [%rl7], %f3;
    ret;
  }


Dissecting the Kernel
---------------------

Now let us dissect the LLVM IR that makes up this kernel.

Data Layout
^^^^^^^^^^^

The data layout string determines the size in bits of common data types, their
ABI alignment, and their storage size.  For NVPTX, you should use one of the
following:

32-bit PTX:

.. code-block:: llvm

  target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

64-bit PTX:

.. code-block:: llvm

  target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"


Target Intrinsics
^^^^^^^^^^^^^^^^^

In this example, we use the ``@llvm.nvvm.read.ptx.sreg.tid.x`` intrinsic to
read the X component of the current thread's ID, which corresponds to a read
of register ``%tid.x`` in PTX. The NVPTX back-end supports a large set of
intrinsics.  A short list is shown below; please see
``include/llvm/IR/IntrinsicsNVVM.td`` for the full list.


================================================ ====================
Intrinsic                                        CUDA Equivalent
================================================ ====================
``i32 @llvm.nvvm.read.ptx.sreg.tid.{x,y,z}``     threadIdx.{x,y,z}
``i32 @llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}``   blockIdx.{x,y,z}
``i32 @llvm.nvvm.read.ptx.sreg.ntid.{x,y,z}``    blockDim.{x,y,z}
``i32 @llvm.nvvm.read.ptx.sreg.nctaid.{x,y,z}``  gridDim.{x,y,z}
``void @llvm.nvvm.barrier0()``                   __syncthreads()
================================================ ====================


Address Spaces
^^^^^^^^^^^^^^

You may have noticed that all of the pointer types in the LLVM IR example had
an explicit address space specifier. What is address space 1? NVIDIA GPU
devices (generally) have four types of memory:

- Global: Large, off-chip memory
- Shared: Small, on-chip memory shared among all threads in a CTA
- Local: Per-thread, private memory
- Constant: Read-only memory shared across all threads

These different types of memory are represented in LLVM IR as address spaces.
There is also a fifth address space used by the NVPTX code generator that
corresponds to the "generic" address space.  This address space can represent
addresses in any other address space (with a few exceptions).  This allows
users to write IR functions that can load/store memory using the same
instructions. Intrinsics are provided to convert pointers between the generic
and non-generic address spaces.

See :ref:`address_spaces` and :ref:`nvptx_intrinsics` for more information.


Kernel Metadata
^^^^^^^^^^^^^^^

In PTX, a function can be either a `kernel` function (callable from the host
program), or a `device` function (callable only from GPU code). You can think
of `kernel` functions as entry-points in the GPU program. To mark an LLVM IR
function as a `kernel` function, we make use of special LLVM metadata. The
NVPTX back-end will look for a named metadata node called
``nvvm.annotations``. This named metadata must contain a list of metadata that
describe the IR. For our purposes, we need to declare a metadata node that
assigns the "kernel" attribute to the LLVM IR function that should be emitted
as a PTX `kernel` function. These metadata nodes take the form:

.. code-block:: text

  !{<function ref>, metadata !"kernel", i32 1}

For the previous example, we have:

.. code-block:: llvm

  !nvvm.annotations = !{!0}
  !0 = !{ptr @kernel, !"kernel", i32 1}

Here, we have a single metadata declaration in ``nvvm.annotations``. This
metadata annotates our ``@kernel`` function with the ``kernel`` attribute.


Running the Kernel
------------------

Generating PTX from LLVM IR is all well and good, but how do we execute it on
a real GPU device? The CUDA Driver API provides a convenient mechanism for
loading and JIT compiling PTX to a native GPU device, and launching a kernel.
The API is similar to OpenCL.  A simple example showing how to load and
execute our vector addition code is shown below. Note that for brevity this
code does not perform much error checking!

.. note::

  You can also use the ``ptxas`` tool provided by the CUDA Toolkit to offline
  compile PTX to machine code (SASS) for a specific GPU architecture. Such
  binaries can be loaded by the CUDA Driver API in the same way as PTX. This
  can be useful for reducing startup time by precompiling the PTX kernels.


.. code-block:: c++

  #include <iostream>
  #include <fstream>
  #include <cassert>
  #include "cuda.h"


  void checkCudaErrors(CUresult err) {
    assert(err == CUDA_SUCCESS);
  }

  /// main - Program entry point
  int main(int argc, char **argv) {
    CUdevice    device;
    CUmodule    cudaModule;
    CUcontext   context;
    CUfunction  function;
    CUlinkState linker;
    int         devCount;

    // CUDA initialization
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&devCount));
    checkCudaErrors(cuDeviceGet(&device, 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));
    std::cout << "Using CUDA Device [0]: " << name << "\n";

    int devMajor, devMinor;
    checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
    std::cout << "Device Compute Capability: "
              << devMajor << "." << devMinor << "\n";
    if (devMajor < 2) {
      std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
      return 1;
    }

    std::ifstream t("kernel.ptx");
    if (!t.is_open()) {
      std::cerr << "kernel.ptx not found\n";
      return 1;
    }
    std::string str((std::istreambuf_iterator<char>(t)),
                      std::istreambuf_iterator<char>());

    // Create driver context
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Create module for object
    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0));

    // Get kernel function
    checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "kernel"));

    // Device data
    CUdeviceptr devBufferA;
    CUdeviceptr devBufferB;
    CUdeviceptr devBufferC;

    checkCudaErrors(cuMemAlloc(&devBufferA, sizeof(float)*16));
    checkCudaErrors(cuMemAlloc(&devBufferB, sizeof(float)*16));
    checkCudaErrors(cuMemAlloc(&devBufferC, sizeof(float)*16));

    float* hostA = new float[16];
    float* hostB = new float[16];
    float* hostC = new float[16];

    // Populate input
    for (unsigned i = 0; i != 16; ++i) {
      hostA[i] = (float)i;
      hostB[i] = (float)(2*i);
      hostC[i] = 0.0f;
    }

    checkCudaErrors(cuMemcpyHtoD(devBufferA, &hostA[0], sizeof(float)*16));
    checkCudaErrors(cuMemcpyHtoD(devBufferB, &hostB[0], sizeof(float)*16));


    unsigned blockSizeX = 16;
    unsigned blockSizeY = 1;
    unsigned blockSizeZ = 1;
    unsigned gridSizeX  = 1;
    unsigned gridSizeY  = 1;
    unsigned gridSizeZ  = 1;

    // Kernel parameters
    void *KernelParams[] = { &devBufferA, &devBufferB, &devBufferC };

    std::cout << "Launching kernel\n";

    // Kernel launch
    checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                   blockSizeX, blockSizeY, blockSizeZ,
                                   0, NULL, KernelParams, NULL));

    // Retrieve device data
    checkCudaErrors(cuMemcpyDtoH(&hostC[0], devBufferC, sizeof(float)*16));


    std::cout << "Results:\n";
    for (unsigned i = 0; i != 16; ++i) {
      std::cout << hostA[i] << " + " << hostB[i] << " = " << hostC[i] << "\n";
    }


    // Clean up after ourselves
    delete [] hostA;
    delete [] hostB;
    delete [] hostC;

    // Clean-up
    checkCudaErrors(cuMemFree(devBufferA));
    checkCudaErrors(cuMemFree(devBufferB));
    checkCudaErrors(cuMemFree(devBufferC));
    checkCudaErrors(cuModuleUnload(cudaModule));
    checkCudaErrors(cuCtxDestroy(context));

    return 0;
  }


You will need to link with the CUDA driver and specify the path to cuda.h.

.. code-block:: text

  # clang++ sample.cpp -o sample -O2 -g -I/usr/local/cuda-5.5/include -lcuda

We don't need to specify a path to ``libcuda.so`` since this is installed in a
system location by the driver, not the CUDA toolkit.

If everything goes as planned, you should see the following output when
running the compiled program:

.. code-block:: text

  Using CUDA Device [0]: GeForce GTX 680
  Device Compute Capability: 3.0
  Launching kernel
  Results:
  0 + 0 = 0
  1 + 2 = 3
  2 + 4 = 6
  3 + 6 = 9
  4 + 8 = 12
  5 + 10 = 15
  6 + 12 = 18
  7 + 14 = 21
  8 + 16 = 24
  9 + 18 = 27
  10 + 20 = 30
  11 + 22 = 33
  12 + 24 = 36
  13 + 26 = 39
  14 + 28 = 42
  15 + 30 = 45

.. note::

  You will likely see a different device identifier based on your hardware


Tutorial: Linking with Libdevice
================================

In this tutorial, we show a simple example of linking LLVM IR with the
libdevice library. We will use the same kernel as the previous tutorial,
except that we will compute ``C = pow(A, B)`` instead of ``C = A + B``.
Libdevice provides an ``__nv_powf`` function that we will use.

.. code-block:: llvm

  target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
  target triple = "nvptx64-nvidia-cuda"

  ; Intrinsic to read X component of thread ID
  declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind
  ; libdevice function
  declare float @__nv_powf(float, float)

  define void @kernel(ptr addrspace(1) %A,
                      ptr addrspace(1) %B,
                      ptr addrspace(1) %C) {
  entry:
    ; What is my ID?
    %id = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind

    ; Compute pointers into A, B, and C
    %ptrA = getelementptr float, ptr addrspace(1) %A, i32 %id
    %ptrB = getelementptr float, ptr addrspace(1) %B, i32 %id
    %ptrC = getelementptr float, ptr addrspace(1) %C, i32 %id

    ; Read A, B
    %valA = load float, ptr addrspace(1) %ptrA, align 4
    %valB = load float, ptr addrspace(1) %ptrB, align 4

    ; Compute C = pow(A, B)
    %valC = call float @__nv_powf(float %valA, float %valB)

    ; Store back to C
    store float %valC, ptr addrspace(1) %ptrC, align 4

    ret void
  }

  !nvvm.annotations = !{!0}
  !0 = !{ptr @kernel, !"kernel", i32 1}


To compile this kernel, we perform the following steps:

1. Link with libdevice
2. Internalize all but the public kernel function
3. Run ``NVVMReflect`` and set ``__CUDA_FTZ`` to 0
4. Optimize the linked module
5. Codegen the module


These steps can be performed by the LLVM ``llvm-link``, ``opt``, and ``llc``
tools. In a complete compiler, these steps can also be performed entirely
programmatically by setting up an appropriate pass configuration (see
:ref:`libdevice`).

.. code-block:: text

  # llvm-link t2.bc libdevice.compute_20.10.bc -o t2.linked.bc
  # opt -internalize -internalize-public-api-list=kernel -nvvm-reflect-list=__CUDA_FTZ=0 -nvvm-reflect -O3 t2.linked.bc -o t2.opt.bc
  # llc -mcpu=sm_20 t2.opt.bc -o t2.ptx

.. note::

  The ``-nvvm-reflect-list=_CUDA_FTZ=0`` is not strictly required, as any
  undefined variables will default to zero. It is shown here for evaluation
  purposes.


This gives us the following PTX (excerpt):

.. code-block:: text

  //
  // Generated by LLVM NVPTX Back-End
  //

  .version 3.1
  .target sm_20
  .address_size 64

    // .globl kernel
                                          // @kernel
  .visible .entry kernel(
    .param .u64 kernel_param_0,
    .param .u64 kernel_param_1,
    .param .u64 kernel_param_2
  )
  {
    .reg .pred  %p<30>;
    .reg .f32   %f<111>;
    .reg .s32   %r<21>;
    .reg .s64   %rl<8>;

  // %bb.0:                                // %entry
    ld.param.u64  %rl2, [kernel_param_0];
    mov.u32   %r3, %tid.x;
    ld.param.u64  %rl3, [kernel_param_1];
    mul.wide.s32  %rl4, %r3, 4;
    add.s64   %rl5, %rl2, %rl4;
    ld.param.u64  %rl6, [kernel_param_2];
    add.s64   %rl7, %rl3, %rl4;
    add.s64   %rl1, %rl6, %rl4;
    ld.global.f32   %f1, [%rl5];
    ld.global.f32   %f2, [%rl7];
    setp.eq.f32 %p1, %f1, 0f3F800000;
    setp.eq.f32 %p2, %f2, 0f00000000;
    or.pred   %p3, %p1, %p2;
    @%p3 bra  BB0_1;
    bra.uni   BB0_2;
  BB0_1:
    mov.f32   %f110, 0f3F800000;
    st.global.f32   [%rl1], %f110;
    ret;
  BB0_2:                                  // %__nv_isnanf.exit.i
    abs.f32   %f4, %f1;
    setp.gtu.f32  %p4, %f4, 0f7F800000;
    @%p4 bra  BB0_4;
  // %bb.3:                                // %__nv_isnanf.exit5.i
    abs.f32   %f5, %f2;
    setp.le.f32 %p5, %f5, 0f7F800000;
    @%p5 bra  BB0_5;
  BB0_4:                                  // %.critedge1.i
    add.f32   %f110, %f1, %f2;
    st.global.f32   [%rl1], %f110;
    ret;
  BB0_5:                                  // %__nv_isinff.exit.i

    ...

  BB0_26:                                 // %__nv_truncf.exit.i.i.i.i.i
    mul.f32   %f90, %f107, 0f3FB8AA3B;
    cvt.rzi.f32.f32 %f91, %f90;
    mov.f32   %f92, 0fBF317200;
    fma.rn.f32  %f93, %f91, %f92, %f107;
    mov.f32   %f94, 0fB5BFBE8E;
    fma.rn.f32  %f95, %f91, %f94, %f93;
    mul.f32   %f89, %f95, 0f3FB8AA3B;
    // inline asm
    ex2.approx.ftz.f32 %f88,%f89;
    // inline asm
    add.f32   %f96, %f91, 0f00000000;
    ex2.approx.f32  %f97, %f96;
    mul.f32   %f98, %f88, %f97;
    setp.lt.f32 %p15, %f107, 0fC2D20000;
    selp.f32  %f99, 0f00000000, %f98, %p15;
    setp.gt.f32 %p16, %f107, 0f42D20000;
    selp.f32  %f110, 0f7F800000, %f99, %p16;
    setp.eq.f32 %p17, %f110, 0f7F800000;
    @%p17 bra   BB0_28;
  // %bb.27:
    fma.rn.f32  %f110, %f110, %f108, %f110;
  BB0_28:                                 // %__internal_accurate_powf.exit.i
    setp.lt.f32 %p18, %f1, 0f00000000;
    setp.eq.f32 %p19, %f3, 0f3F800000;
    and.pred    %p20, %p18, %p19;
    @!%p20 bra  BB0_30;
    bra.uni   BB0_29;
  BB0_29:
    mov.b32    %r9, %f110;
    xor.b32   %r10, %r9, -2147483648;
    mov.b32    %f110, %r10;
  BB0_30:                                 // %__nv_powf.exit
    st.global.f32   [%rl1], %f110;
    ret;
  }
