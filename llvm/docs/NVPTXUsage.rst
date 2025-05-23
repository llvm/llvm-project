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
code. By default, the back-end will emit device functions. The ``ptx_kernel``
calling convention is used to declare a function as a kernel function.

The following example shows a kernel function calling a device function in LLVM
IR. The function ``@my_kernel`` is callable from host code, but ``@my_fmad`` is
not.

.. code-block:: llvm

    define float @my_fmad(float %x, float %y, float %z) {
      %mul = fmul float %x, %y
      %add = fadd float %mul, %z
      ret float %add
    }

    define ptx_kernel void @my_kernel(ptr %ptr) {
      %val = load float, ptr %ptr
      %ret = call float @my_fmad(float %val, float %val, float %val)
      store float %ret, ptr %ptr
      ret void
    }

When compiled, the PTX kernel functions are callable by host-side code.

.. _nvptx_fnattrs:

Function Attributes
-------------------

``"nvvm.maxclusterrank"="<n>"``
    This attribute specifies the maximum number of blocks per cluster. Must be 
    non-zero. Only supported for Hopper+.

``"nvvm.minctasm"="<n>"``
    This indicates a hint/directive to the compiler/driver, asking it to put at
    least these many CTAs on an SM.

``"nvvm.maxnreg"="<n>"``
    This attribute indicates the maximum number of registers to be used for the
    kernel function.

``"nvvm.maxntid"="<x>[,<y>[,<z>]]"``
    This attribute declares the maximum number of threads in the thread block
    (CTA). The maximum number of threads is the product of the maximum extent in
    each dimension. Exceeding the maximum number of threads results in a runtime
    error or kernel launch failure.

``"nvvm.reqntid"="<x>[,<y>[,<z>]]"``
    This attribute declares the exact number of threads in the thread block
    (CTA). The number of threads is the product of the value in each dimension.
    Specifying a different CTA dimension at launch will result in a runtime 
    error or kernel launch failure.

``"nvvm.cluster_dim"="<x>[,<y>[,<z>]]"``
    This attribute declares the number of thread blocks (CTAs) in the cluster.
    The total number of CTAs is the product of the number of CTAs in each 
    dimension. Specifying a different cluster dimension at launch will result in
    a runtime error or kernel launch failure. Only supported for Hopper+.

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
   7             Shared Cluster
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

'``llvm.nvvm.mapa.*``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare ptr @llvm.nvvm.mapa(ptr %p, i32 %rank)
    declare ptr addrspace(7) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3) %p, i32 %rank)

Overview:
"""""""""

The '``llvm.nvvm.mapa.*``' intrinsics map a shared memory pointer ``p`` of another CTA with ``%rank`` to the current CTA.
The ``llvm.nvvm.mapa`` form expects a generic pointer to shared memory and returns a generic pointer to shared cluster memory.
The ``llvm.nvvm.mapa.shared.cluster`` form expects a pointer to shared memory and returns a pointer to shared cluster memory.
They corresponds directly to the ``mapa`` and ``mapa.shared.cluster`` PTX instructions.

Semantics:
""""""""""

If the given pointer in the generic address space refers to memory which falls
within the state space of the intrinsic (and therefore could be safely address
space casted to this space), 1 is returned, otherwise 0 is returned.

Arithmetic Intrinsics
---------------------

'``llvm.nvvm.fabs.*``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare float @llvm.nvvm.fabs.f32(float %a)
    declare double @llvm.nvvm.fabs.f64(double %a)
    declare half @llvm.nvvm.fabs.f16(half %a)
    declare <2 x half> @llvm.nvvm.fabs.v2f16(<2 x half> %a)
    declare bfloat @llvm.nvvm.fabs.bf16(bfloat %a)
    declare <2 x bfloat> @llvm.nvvm.fabs.v2bf16(<2 x bfloat> %a)

Overview:
"""""""""

The '``llvm.nvvm.fabs.*``' intrinsics return the absolute value of the operand.

Semantics:
""""""""""

Unlike, '``llvm.fabs.*``', these intrinsics do not perfectly preserve NaN
values. Instead, a NaN input yeilds an unspecified NaN output.


'``llvm.nvvm.fabs.ftz.*``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare float @llvm.nvvm.fabs.ftz.f32(float %a)
    declare half @llvm.nvvm.fabs.ftz.f16(half %a)
    declare <2 x half> @llvm.nvvm.fabs.ftz.v2f16(<2 x half> %a)

Overview:
"""""""""

The '``llvm.nvvm.fabs.ftz.*``' intrinsics return the absolute value of the
operand, flushing subnormals to sign preserving zero.

Semantics:
""""""""""

Before the absolute value is taken, the input is flushed to sign preserving
zero if it is a subnormal. In addition, unlike '``llvm.fabs.*``', a NaN input
yields an unspecified NaN output.


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
shift amount is clamped at the integer width (instead of modulo it). Currently,
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
shift amount is clamped at the integer width (instead of modulo it). Currently,
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

'``llvm.nvvm.{zext,sext}.{wrap,clamp}``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.zext.wrap(i32 %a, i32 %b)
    declare i32 @llvm.nvvm.zext.clamp(i32 %a, i32 %b)
    declare i32 @llvm.nvvm.sext.wrap(i32 %a, i32 %b)
    declare i32 @llvm.nvvm.sext.clamp(i32 %a, i32 %b)

Overview:
"""""""""

The '``llvm.nvvm.{zext,sext}.{wrap,clamp}``' family of intrinsics extracts the
low bits of the input value, and zero- or sign-extends them back to the original
width.

Semantics:
""""""""""

The '``llvm.nvvm.{zext,sext}.{wrap,clamp}``' family of intrinsics returns
extension of N lowest bits of operand %a. For the '``wrap``' variants, N is the
value of operand %b modulo 32. For the '``clamp``' variants, N is the value of
operand %b clamped to the range [0, 32]. The N lowest bits are then
zero-extended the case of the '``zext``' variants, or sign-extended the case of
the '``sext``' variants. If N is 0, the result is 0.

'``llvm.nvvm.bmsk.{wrap,clamp}``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.bmsk.wrap(i32 %a, i32 %b)
    declare i32 @llvm.nvvm.bmsk.clamp(i32 %a, i32 %b)

Overview:
"""""""""

The '``llvm.nvvm.bmsk.{wrap,clamp}``' family of intrinsics creates a bit mask
given a starting bit position and a bit width.

Semantics:
""""""""""

The '``llvm.nvvm.bmsk.{wrap,clamp}``' family of intrinsics returns a value with
all bits set to 0 except for %b bits starting at bit position %a. For the
'``wrap``' variants, the values of %a and %b modulo 32 are used. For the
'``clamp``' variants, the values of %a and %b are clamped to the range [0, 32],
which in practice is equivalent to using them as is.

'``llvm.nvvm.prmt``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.prmt(i32 %lo, i32 %hi, i32 %selector)

Overview:
"""""""""

The '``llvm.nvvm.prmt``' constructs a permutation of the bytes of the first two
operands, selecting based on the third operand.

Semantics:
""""""""""

The bytes in the first two source operands are numbered from 0 to 7:
{%hi, %lo} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}. For each byte in the target
register, a 4-bit selection value is defined.

The 3 lsbs of the selection value specify which of the 8 source bytes should be
moved into the target position. The msb defines if the byte value should be
copied, or if the sign (msb of the byte) should be replicated over all 8 bits
of the target position (sign extend of the byte value); msb=0 means copy the
literal value; msb=1 means replicate the sign.

These 4-bit selection values are pulled from the lower 16-bits of the %selector
operand, with the least significant selection value corresponding to the least
significant byte of the destination.


'``llvm.nvvm.prmt.*``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

    declare i32 @llvm.nvvm.prmt.f4e(i32 %lo, i32 %hi, i32 %selector)
    declare i32 @llvm.nvvm.prmt.b4e(i32 %lo, i32 %hi, i32 %selector)

    declare i32 @llvm.nvvm.prmt.rc8(i32 %lo, i32 %selector)
    declare i32 @llvm.nvvm.prmt.ecl(i32 %lo, i32 %selector)
    declare i32 @llvm.nvvm.prmt.ecr(i32 %lo, i32 %selector)
    declare i32 @llvm.nvvm.prmt.rc16(i32 %lo, i32 %selector)

Overview:
"""""""""

The '``llvm.nvvm.prmt.*``' family of intrinsics constructs a permutation of the
bytes of the first one or two operands, selecting based on the 2 least
significant bits of the final operand.

Semantics:
""""""""""

As with the generic '``llvm.nvvm.prmt``' intrinsic, the bytes in the first one
or two source operands are numbered. The first source operand (%lo) is numbered
{b3, b2, b1, b0}, in the case of the '``f4e``' and '``b4e``' variants, the
second source operand (%hi) is numbered {b7, b6, b5, b4}.

Depending on the 2 least significant bits of the %selector operand, the result
of the permutation is defined as follows:

+------------+----------------+--------------+
|    Mode    | %selector[1:0] |    Output    |
+------------+----------------+--------------+
| '``f4e``'  | 0              | {3, 2, 1, 0} |
|            +----------------+--------------+
|            | 1              | {4, 3, 2, 1} |
|            +----------------+--------------+
|            | 2              | {5, 4, 3, 2} |
|            +----------------+--------------+
|            | 3              | {6, 5, 4, 3} |
+------------+----------------+--------------+
| '``b4e``'  | 0              | {5, 6, 7, 0} |
|            +----------------+--------------+
|            | 1              | {6, 7, 0, 1} |
|            +----------------+--------------+
|            | 2              | {7, 0, 1, 2} |
|            +----------------+--------------+
|            | 3              | {0, 1, 2, 3} |
+------------+----------------+--------------+
| '``rc8``'  | 0              | {0, 0, 0, 0} |
|            +----------------+--------------+
|            | 1              | {1, 1, 1, 1} |
|            +----------------+--------------+
|            | 2              | {2, 2, 2, 2} |
|            +----------------+--------------+
|            | 3              | {3, 3, 3, 3} |
+------------+----------------+--------------+
| '``ecl``'  | 0              | {3, 2, 1, 0} |
|            +----------------+--------------+
|            | 1              | {3, 2, 1, 1} |
|            +----------------+--------------+
|            | 2              | {3, 2, 2, 2} |
|            +----------------+--------------+
|            | 3              | {3, 3, 3, 3} |
+------------+----------------+--------------+
| '``ecr``'  | 0              | {0, 0, 0, 0} |
|            +----------------+--------------+
|            | 1              | {1, 1, 1, 0} |
|            +----------------+--------------+
|            | 2              | {2, 2, 1, 0} |
|            +----------------+--------------+
|            | 3              | {3, 2, 1, 0} |
+------------+----------------+--------------+
| '``rc16``' | 0              | {1, 0, 1, 0} |
|            +----------------+--------------+
|            | 1              | {3, 2, 3, 2} |
|            +----------------+--------------+
|            | 2              | {1, 0, 1, 0} |
|            +----------------+--------------+
|            | 3              | {3, 2, 3, 2} |
+------------+----------------+--------------+

TMA family of Intrinsics
------------------------

'``llvm.nvvm.cp.async.bulk.global.to.shared.cluster``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(7) %dst, ptr addrspace(3) %mbar, ptr addrspace(1) %src, i32 %size, i16 %mc, i64 %ch, i1 %flag_mc, i1 %flag_ch)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.global.to.shared.cluster``' intrinsic
corresponds to the ``cp.async.bulk.shared::cluster.global.*`` family
of PTX instructions. These instructions initiate an asynchronous
copy of bulk data from global memory to shared::cluster memory.
The 32-bit operand ``%size`` specifies the amount of memory to be
copied and it must be a multiple of 16.

* The last two arguments to these intrinsics are boolean flags
  indicating support for cache_hint and/or multicast modifiers.
  These flag arguments must be compile-time constants. The backend
  looks through these flags and lowers the intrinsics appropriately.

* The Nth argument (denoted by ``i1 %flag_ch``) when set, indicates
  a valid cache_hint (``i64 %ch``) and generates the ``.L2::cache_hint``
  variant of the PTX instruction.

* The [N-1]th argument (denoted by ``i1 %flag_mc``) when set, indicates
  the presence of a multicast mask (``i16 %mc``) and generates the PTX
  instruction with the ``.multicast::cluster`` modifier.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`_.

'``llvm.nvvm.cp.async.bulk.shared.cta.to.global``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.shared.cta.to.global(ptr addrspace(1) %dst, ptr addrspace(3) %src, i32 %size, i64 %ch, i1 %flag_ch)
  declare void @llvm.nvvm.cp.async.bulk.shared.cta.to.global.bytemask(..., i32 %size, i64 %ch, i1 %flag_ch, i16 %mask)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.shared.cta.to.global``' intrinsic
corresponds to the ``cp.async.bulk.global.shared::cta.*`` set of PTX
instructions. These instructions initiate an asynchronous copy from
shared::cta to global memory. The 32-bit operand ``%size`` specifies
the amount of memory to be copied (in bytes) and it must be a multiple
of 16. For the ``.bytemask`` variant, the 16-bit wide mask operand
specifies whether the i-th byte of each 16-byte wide chunk of source
data is copied to the destination.

* The ``i1 %flag_ch`` argument to these intrinsics is a boolean
  flag indicating support for cache_hint. This flag argument must
  be a compile-time constant. When set, it indicates a valid
  cache_hint (``i64 %ch``) and generates the ``.L2::cache_hint``
  variant of the PTX instruction.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`_.

'``llvm.nvvm.cp.async.bulk.shared.cta.to.cluster``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.shared.cta.to.cluster(ptr addrspace(7) %dst, ptr addrspace(3) %mbar, ptr addrspace(3) %src, i32 %size)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.shared.cta.to.cluster``' intrinsic
corresponds to the ``cp.async.bulk.shared::cluster.shared::cta.*``
PTX instruction. This instruction initiates an asynchronous copy from
shared::cta to shared::cluster memory. The destination has to be in
the shared memory of a different CTA within the cluster. The 32-bit
operand ``%size`` specifies the amount of memory to be copied and
it must be a multiple of 16.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`_.

'``llvm.nvvm.cp.async.bulk.prefetch.L2``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.prefetch.L2(ptr addrspace(1) %src, i32 %size, i64 %ch, i1 %flag_ch)

Overview:
"""""""""

The '``@llvm.nvvm.cp.async.bulk.prefetch.L2``' intrinsic
corresponds to the ``cp.async.bulk.prefetch.L2.*`` family
of PTX instructions. These instructions initiate an asynchronous
prefetch of bulk data from global memory to the L2 cache.
The 32-bit operand ``%size`` specifies the amount of memory to be
prefetched in terms of bytes and it must be a multiple of 16.

* The last argument to these intrinsics is boolean flag indicating
  support for cache_hint. These flag argument must be compile-time
  constant. When set, it indicates a valid cache_hint (``i64 %ch``)
  and generates the ``.L2::cache_hint`` variant of the PTX instruction.

For more information, refer PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch>`_.

'``llvm.nvvm.prefetch.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void  @llvm.nvvm.prefetch.global.L1(ptr addrspace(1) %global_ptr)
  declare void  @llvm.nvvm.prefetch.global.L2(ptr addrspace(1) %global_ptr)
  declare void  @llvm.nvvm.prefetch.local.L1(ptr addrspace(5) %local_ptr)
  declare void  @llvm.nvvm.prefetch.local.L2(ptr addrspace(5) %local_ptr)
  
  declare void  @llvm.nvvm.prefetch.L1(ptr %ptr)
  declare void  @llvm.nvvm.prefetch.L2(ptr %ptr)
  
  declare void  @llvm.nvvm.prefetch.global.L2.evict.normal(ptr addrspace(1) %global_ptr)
  declare void  @llvm.nvvm.prefetch.global.L2.evict.last(ptr addrspace(1) %global_ptr)

  declare void  @llvm.nvvm.prefetchu.L1(ptr %ptr)

Overview:
"""""""""

The '``@llvm.nvvm.prefetch.*``' and '``@llvm.nvvm.prefetchu.*``' intrinsic
correspond to the '``prefetch.*``;' and '``prefetchu.*``' family of PTX instructions. 
The '``prefetch.*``' instructions bring the cache line containing the
specified address in global or local memory address space into the 
specified cache level (L1 or L2). The '`prefetchu.*``' instruction brings the cache line 
containing the specified generic address into the specified uniform cache level.
If no address space is specified, it is assumed to be generic address. The intrinsic 
uses and eviction priority which can be accessed by the '``.level::eviction_priority``' modifier.

* A prefetch to a shared memory location performs no operation.
* A prefetch into the uniform cache requires a generic address, 
  and no operation occurs if the address maps to a const, local, or shared memory location.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu>`_.

'``llvm.nvvm.applypriority.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void  @llvm.nvvm.applypriority.global.L2.evict.normal(ptr addrspace(1) %global_ptr, i64 %size)
  declare void  @llvm.nvvm.applypriority.L2.evict.normal(ptr %ptr, i64 %size)

Overview:
"""""""""

The '``@llvm.nvvm.applypriority.*``'  applies the cache eviction priority specified by the
.level::eviction_priority qualifier to the address range [a..a+size) in the specified cache 
level. If no state space is specified then Generic Addressing is used. If the specified address 
does not fall within the address window of .global state space then the behavior is undefined.
The operand size is an integer constant that specifies the amount of data, in bytes, in the specified cache
level on which the priority is to be applied. The only supported value for the size operand is 128.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-applypriority>`_.

``llvm.nvvm.discard.*``'
^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void  @llvm.nvvm.discard.global.L2(ptr addrspace(1) %global_ptr, i64 immarg)
  declare void  @llvm.nvvm.discard.L2(ptr %ptr, i64 immarg)

Overview:
"""""""""

The *effects* of the ``@llvm.nvvm.discard.L2*`` intrinsics are those of a non-atomic 
non-volatile ``llvm.memset`` that writes ``undef`` to the destination 
address range ``[%ptr, %ptr + immarg)``. The ``%ptr`` must be aligned by 128 bytes.
Subsequent reads from the address range may read ``undef`` until the memory is overwritten 
with a different value.
These operations *hint* the implementation that data in the L2 cache can be destructively 
discarded without writing it back to memory. 
The operand ``immarg`` is an integer constant that specifies the length in bytes of the 
address range ``[%ptr, %ptr + immarg)`` to write ``undef`` into. 
The only supported value for the ``immarg`` operand is ``128``. 
If generic addressing is used and the specified address does not fall within the 
address window of global memory (``addrspace(1)``) the behavior is undefined.

.. code-block:: llvm
 
   call void @llvm.nvvm.discard.L2(ptr %p, i64 128)  ;; writes `undef` to [p, p+128)
   %a = load i64, ptr %p. ;; loads 8 bytes containing undef
   %b = load i64, ptr %p  ;; loads 8 bytes containing undef
   ;; comparing %a and %b compares `undef` values!
   %fa = freeze i64 %a  ;; freezes undef to stable bit-pattern
   %fb = freeze i64 %b  ;; freezes undef to stable bit-pattern
   ;; %fa may compare different to %fb!
   
For more information, refer to the  `CUDA C++ discard documentation <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/discard_memory.html>`__ and to the `PTX ISA discard documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-discard>`__ .

'``llvm.nvvm.cp.async.bulk.tensor.g2s.tile.[1-5]d``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %dst, ptr addrspace(3) %bar, ptr %tensor_map, i32 %d0, i16 %mc, i64 %ch, i1 %flag_mc, i1 %flag_ch)
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

'``llvm.nvvm.griddepcontrol.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.griddepcontrol.launch_dependents()
  declare void @llvm.nvvm.griddepcontrol.wait()

Overview:
"""""""""

The ``griddepcontrol`` intrinsics allows the dependent grids and prerequisite grids as defined by the runtime, to control execution in the following way:

``griddepcontrol.launch_dependents`` intrinsic signals that the dependents can be scheduled, before the current grid completes. The intrinsic can be invoked by multiple threads in the current CTA and repeated invocations of the intrinsic will have no additional side effects past that of the first invocation.

``griddepcontrol.wait`` intrinsic causes the executing thread to wait until all prerequisite grids in flight have completed and all the memory operations from the prerequisite grids are performed and made visible to the current grid.

For more information, refer 
`PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol>`__.

TCGEN05 family of Intrinsics
----------------------------

The llvm.nvvm.tcgen05.* intrinsics model the TCGEN05 family of instructions
exposed by PTX. These intrinsics use 'Tensor Memory' (henceforth ``tmem``).
NVPTX represents this memory using ``addrspace(6)`` and is always 32-bits.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory>`_.

The tensor-memory pointers may only be used with the tcgen05 intrinsics.
There are specialized load/store instructions provided (tcgen05.ld/st) to
work with tensor-memory.

See the PTX ISA for more information on tensor-memory load/store instructions
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-and-register-load-store-instructions>`_.

'``llvm.nvvm.tcgen05.alloc``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.alloc.cg1(ptr %dst, i32 %ncols)
  declare void @llvm.nvvm.tcgen05.alloc.cg2(ptr %dst, i32 %ncols)
  declare void @llvm.nvvm.tcgen05.alloc.shared.cg1(ptr addrspace(3) %dst, i32 %ncols)
  declare void @llvm.nvvm.tcgen05.alloc.shared.cg2(ptr addrspace(3) %dst, i32 %ncols)

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.alloc.*``' intrinsics correspond to the
``tcgen05.alloc.cta_group*.sync.aligned.b32`` family of PTX instructions.
The ``tcgen05.alloc`` is a potentially blocking instruction which dynamically
allocates the specified number of columns in the Tensor Memory and writes
the address of the allocated Tensor Memory into shared memory at the
location specified by ``%dst``. The 32-bit operand ``%ncols`` specifies
the number of columns to be allocated and it must be a power-of-two.
The ``.shared`` variant explicitly uses shared memory address space for
the ``%dst`` operand. The ``.cg1`` and ``.cg2`` variants generate
``cta_group::1`` and ``cta_group::2`` variants of the instruction respectively.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-allocation-and-management-instructions>`_.

'``llvm.nvvm.tcgen05.dealloc``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.dealloc.cg1(ptr addrspace(6) %tmem_addr, i32 %ncols)
  declare void @llvm.nvvm.tcgen05.dealloc.cg2(ptr addrspace(6) %tmem_addr, i32 %ncols)

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.dealloc.*``' intrinsics correspond to the
``tcgen05.dealloc.*`` set of PTX instructions. The ``tcgen05.dealloc``
instructions deallocates the Tensor Memory specified by the Tensor Memory
address ``%tmem_addr``. The operand ``%tmem_addr`` must point to a previous
Tensor Memory allocation. The 32-bit operand ``%ncols`` specifies the number
of columns to be de-allocated. The ``.cg1`` and ``.cg2`` variants generate
``cta_group::1`` and ``cta_group::2`` variants of the instruction respectively.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-allocation-and-management-instructions>`_.

'``llvm.nvvm.tcgen05.relinq.alloc.permit``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.relinq.alloc.permit.cg1()
  declare void @llvm.nvvm.tcgen05.relinq.alloc.permit.cg2()

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.relinq.alloc.permit.*``' intrinsics correspond
to the ``tcgen05.relinquish_alloc_permit.*`` set of PTX instructions.
This instruction specifies that the CTA of the executing thread is
relinquishing the right to allocate Tensor Memory. So, it is illegal
for a CTA to perform ``tcgen05.alloc`` after any of its constituent
threads execute ``tcgen05.relinquish_alloc_permit``. The ``.cg1``
and ``.cg2`` variants generate ``cta_group::1`` and ``cta_group::2``
flavors of the instruction respectively.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-allocation-and-management-instructions>`_.

'``llvm.nvvm.tcgen05.commit``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.commit.{cg1,cg2}(ptr %mbar)
  declare void @llvm.nvvm.tcgen05.commit.shared.{cg1,cg2}(ptr addrspace(3) %mbar)
  declare void @llvm.nvvm.tcgen05.commit.mc.{cg1,cg2}(ptr %mbar, i16 %mc)
  declare void @llvm.nvvm.tcgen05.commit.mc.shared.{cg1,cg2}(ptr addrspace(3) %mbar, i16 %mc)

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.commit.*``' intrinsics correspond to the
``tcgen05.commit.{cg1/cg2}.mbarrier::arrive::one.*`` set of PTX instructions.
The ``tcgen05.commit`` is an asynchronous instruction which makes the mbarrier
object (``%mbar``) track the completion of all prior asynchronous tcgen05 operations.
The ``.mc`` variants allow signaling on the mbarrier objects of multiple CTAs
(specified by ``%mc``) in the cluster. The ``.cg1`` and ``.cg2`` variants generate
``cta_group::1`` and ``cta_group::2`` flavors of the instruction respectively.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit>`_.

'``llvm.nvvm.tcgen05.wait``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.wait.ld()
  declare void @llvm.nvvm.tcgen05.wait.st()

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.wait.ld/st``' intrinsics correspond to
the ``tcgen05.wait::{ld/st}.sync.aligned`` pair of PTX instructions.
The ``tcgen05.wait::ld`` causes the executing thread to block until
all prior ``tcgen05.ld`` operations issued by the executing thread
have completed. The ``tcgen05.wait::st`` causes the executing thread
to block until all prior ``tcgen05.st`` operations issued by the
executing thread have completed.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait>`_.

'``llvm.nvvm.tcgen05.fence``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.fence.before.thread.sync()
  declare void @llvm.nvvm.tcgen05.fence.after.thread.sync()

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.fence.*``' intrinsics correspond to
the ``tcgen05.fence::{before/after}_thread_sync`` pair of PTX instructions.
These instructions act as code motion fences for asynchronous tcgen05
operations.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-fence>`_.

'``llvm.nvvm.tcgen05.shift``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.shift.down.cg1(ptr addrspace(6) %tmem_addr)
  declare void @llvm.nvvm.tcgen05.shift.down.cg2(ptr addrspace(6) %tmem_addr)

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.shift.{cg1/cg2}``' intrinsics correspond to
the ``tcgen05.shift.{cg1/cg2}`` PTX instructions. The ``tcgen05.shift``
is an asynchronous instruction which initiates the shifting of 32-byte
elements downwards across all the rows, except the last, by one row.
The address operand ``%tmem_addr`` specifies the base address of the
matrix in the Tensor Memory whose rows must be down shifted.

For more information, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift>`_.

'``llvm.nvvm.tcgen05.cp``'
^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.cp.4x256b.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.128x256b.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.128x128b.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.32x128b_warpx4.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_02_13.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_01_23.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)

  declare void @llvm.nvvm.tcgen05.cp.4x256b.b6x16_p32.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.128x256b.b6x16_p32.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.128x128b.b6x16_p32.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.32x128b_warpx4.b6x16_p32.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_02_13.b6x16_p32.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_01_23.b6x16_p32.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)

  declare void @llvm.nvvm.tcgen05.cp.4x256b.b4x16_p64.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.128x256b.b4x16_p64.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.128x128b.b4x16_p64.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.32x128b_warpx4.b4x16_p64.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_02_13.b4x16_p64.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)
  declare void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_01_23.b4x16_p64.{cg1,cg2}(ptr addrspace(6) %tmem_addr, i64 %sdesc)

Overview:
"""""""""

The '``@llvm.nvvm.tcgen05.cp.{shape}.{src_fmt}.{cg1/cg2}``' intrinsics
correspond to the ``tcgen05.cp.*`` family of PTX instructions.
The ``tcgen05.cp`` instruction initiates an asynchronous copy operation from
shared memory to the location specified by ``%tmem_addr`` in Tensor Memory.
The 64-bit register operand ``%sdesc`` is the matrix descriptor representing
the source matrix in shared memory that needs to be copied.

The valid shapes for the copy operation are:
{128x256b, 4x256b, 128x128b, 64x128b_warpx2_02_13, 64x128b_warpx2_01_23, 32x128b_warpx4}.

Shapes ``64x128b`` and ``32x128b`` require dedicated multicast qualifiers,
which are appended to the corresponding intrinsic names.

Optionally, the data can be decompressed from the source format in the shared memory
to the destination format in Tensor Memory during the copy operation. Currently,
only ``.b8x16`` is supported as destination format. The valid source formats are
``.b6x16_p32`` and ``.b4x16_p64``.

When the source format is ``.b6x16_p32``, a contiguous set of 16 elements of 6-bits
each followed by four bytes of padding (``_p32``) in shared memory is decompressed
into 16 elements of 8-bits (``.b8x16``) each in the Tensor Memory.

When the source format is ``.b4x16_p64``, a contiguous set of 16 elements of 4-bits
each followed by eight bytes of padding (``_p64``) in shared memory is decompressed
into 16 elements of 8-bits (``.b8x16``) each in the Tensor Memory.

For more information on the decompression schemes, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#optional-decompression>`_.

For more information on the tcgen05.cp instruction, refer to the PTX ISA
`<https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-cp>`_.

'``llvm.nvvm.tcgen05.ld.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare <n x i32> @llvm.nvvm.tcgen05.ld.<shape>.<num>(ptr addrspace(6) %tmem_addr, i1 %pack)

  declare <n x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.<num>(ptr addrspace(6) %tmem_addr, i64 %offset, i1 %pack)

Overview:
"""""""""

This group of intrinsics asynchronously load data from the Tensor Memory at the location specified
by the 32-bit address operand `tmem_addr` into the destination registers, collectively across all threads
of the warps.

All the threads in the warp must specify the same value of `tmem_addr`, which must be the base address
of the collective load operation. Otherwise, the behavior is undefined.

The `shape` qualifier and the `num` qualifier together determines the total dimension of the data ('n') which
is loaded from the Tensor Memory. The `shape` qualifier indicates the base dimension of data. The `num` qualifier
indicates the repeat factor on the base dimension resulting in the total dimension of the data that is accessed.

Allowed values for the 'num' are `x1, x2, x4, x8, x16, x32, x64, x128`.

Allowed values for the 'shape' in the first intrinsic are `16x64b, 16x128b, 16x256b, 32x32b`.

Allowed value for the 'shape' in the second intrinsic is `16x32bx2`.

The result of the intrinsic is a vector consisting of one or more 32-bit registers derived from `shape` and
`num` as shown below.

=========== =========================  ==========  ==========
 num/shape     16x32bx2/16x64b/32x32b    16x128b    16x256b
=========== =========================  ==========  ==========
 x1                 1                      2           4
 x2                 2                      4           8
 x4                 4                      8           16
 x8                 8                      16          32
 x16                16                     32          64
 x32                32                     64          128
 x64                64                     128         NA
 x128               128                    NA          NA
=========== =========================  ==========  ==========

The last argument `i1 %pack` is a compile-time constant which when set, indicates that the adjacent columns are packed into a single 32-bit element during the load

For more information, refer to the
`PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld>`__.


'``llvm.nvvm.tcgen05.st.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.tcgen05.st.<shape>.<num>(ptr addrspace(6) %tmem_addr, <n x i32> %args, i1 %unpack)

  declare void @llvm.nvvm.tcgen05.st.16x32bx2.<num>(ptr addrspace(6) %tmem_addr, <n x i32> %args, i64 %offset, i1 %unpack)

Overview:
"""""""""

This group of intrinsics asynchronously store data from the source vector into the Tensor Memory at the location
specified by the 32-bit address operand 'tmem_addr` collectively across all threads of the warps.

All the threads in the warp must specify the same value of `tmem_addr`, which must be the base address of the
collective load operation. Otherwise, the behavior is undefined.

The `shape` qualifier and the `num` qualifier together determines the total dimension of the data ('n') which
is loaded from the Tensor Memory. The `shape` qualifier indicates the base dimension of data. The `num` qualifier
indicates the repeat factor on the base dimension resulting in the total dimension of the data that is accessed.

Allowed values for the 'num' are `x1, x2, x4, x8, x16, x32, x64, x128`.

Allowed values for the 'shape' in the first intrinsic are `16x64b, 16x128b, 16x256b, 32x32b`.

Allowed value for the 'shape' in the second intrinsic is `16x32bx2`.

`args` argument is a vector consisting of one or more 32-bit registers derived from `shape` and
`num` as listed in the table listed in the `tcgen05.ld` section.

Each shape support an `unpack` mode to allow a 32-bit element in the register to be unpacked into two 16-bit elements and store them in adjacent columns. `unpack` mode can be enabled by setting the `%unpack` operand to 1 and can be disabled by setting it to 0.

The last argument `i1 %unpack` is a compile-time constant which when set, indicates that a 32-bit element in the register to be unpacked into two 16-bit elements and store them in adjacent columns.

For more information, refer to the
`PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st>`__.

Store Intrinsics
----------------

'``llvm.nvvm.st.bulk.*``'
^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.st.bulk(ptr addrspace(1) %dst, i64 %size, i64 immarg %initval)
  declare void @llvm.nvvm.st.bulk.shared.cta(ptr addrspace(3) %dst, i64 %size, i64 immarg %initval)

Overview:
"""""""""

The '``@llvm.nvvm.st.bulk.*``' intrinsics initialize a region of shared memory 
starting from the location specified by the destination address operand `%dst`.

The integer operand `%size` specifies the amount of memory to be initialized in 
terms of number of bytes and must be a multiple of 8. Otherwise, the behavior 
is undefined.

The integer immediate operand `%initval` specifies the initialization value for 
the memory locations. The only numeric value allowed is 0.

The ``@llvm.nvvm.st.bulk.shared.cta`` and ``@llvm.nvvm.st.bulk`` intrinsics are 
similar but the latter uses generic addressing (see `Generic Addressing <https://docs.nvidia.com/cuda/parallel-thread-execution/#generic-addressing>`__).

For more information, refer `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk>`__.


clusterlaunchcontrol Intrinsics
-------------------------------

'``llvm.nvvm.clusterlaunchcontrol.try_cancel*``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare void @llvm.nvvm.clusterlaunchcontrol.try_cancel.async.shared(ptr addrspace(3) %addr, ptr addrspace(3) %mbar)
  declare void @llvm.nvvm.clusterlaunchcontrol.try_cancel.async.multicast.shared(ptr addrspace(3) %addr, ptr addrspace(3) %mbar)

Overview:
"""""""""

The ``clusterlaunchcontrol.try_cancel`` intrinsics requests atomically cancelling
the launch of a cluster that has not started running yet. It asynchronously non-atomically writes
a 16-byte opaque response to shared memory, pointed to by 16-byte-aligned ``addr`` indicating whether the
operation succeeded or failed. ``addr`` and 8-byte-aligned ``mbar`` must refer to ``shared::cta``
otherwise the behavior is undefined. The completion of the asynchronous operation
is tracked using the mbarrier completion mechanism at ``.cluster`` scope referenced
by the shared memory pointer, ``mbar``. On success, the opaque response contains
the CTA id of the first CTA of the canceled cluster; no other successful response
from other ``clusterlaunchcontrol.try_cancel`` operations from the same grid will
contain that id.

The ``multicast`` variant specifies that the response is asynchronously non-atomically written to
the corresponding shared memory location of each CTA in the requesting cluster.
The completion of the write of each local response is tracked by independent
mbarriers at the corresponding shared memory location of each CTA in the
cluster.

For more information, refer `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/?a#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel>`__.

'``llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled``' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare i1 @llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled(i128 %try_cancel_response)

Overview:
"""""""""

The ``llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled`` intrinsic decodes the opaque response written by the
``llvm.nvvm.clusterlaunchcontrol.try_cancel`` operation.

The intrinsic returns ``0`` (false) if the request failed. If the request succeeded,
it returns ``1`` (true). A true result indicates that:

- the thread block cluster whose first CTA id matches that of the response
  handle will not run, and
- no other successful response of another ``try_cancel`` request in the grid will contain
  the first CTA id of that cluster

For more information, refer `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/?a#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel>`__.


'``llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.*``' Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

.. code-block:: llvm

  declare i32 @llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x(i128 %try_cancel_response)
  declare i32 @llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y(i128 %try_cancel_response)
  declare i32 @llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z(i128 %try_cancel_response)

Overview:
"""""""""

The ``clusterlaunchcontrol.query_cancel.get_first_ctaid.*`` intrinsic can be
used to decode the successful opaque response written by the
``llvm.nvvm.clusterlaunchcontrol.try_cancel`` operation.

If the request succeeded:

- ``llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.{x,y,z}`` returns
  the coordinate of the first CTA in the canceled cluster, either x, y, or z.

If the request failed, the behavior of these intrinsics is undefined.

For more information, refer `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/?a#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel>`__.

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
