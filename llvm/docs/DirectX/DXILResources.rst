======================
DXIL Resource Handling
======================

.. contents::
   :local:

.. toctree::
   :hidden:

Introduction
============

Resources in DXIL are represented via ``TargetExtType`` in LLVM IR and
eventually lowered by the DirectX backend into metadata in DXIL.

In DXC and DXIL, static resources are represented as lists of SRVs (Shader
Resource Views), UAVs (Uniform Access Views), CBVs (Constant Bffer Views), and
Samplers. This metadata consists of a "resource record ID" which uniquely
identifies a resource and type information. As of shader model 6.6, there are
also dynamic resources, which forgo the metadata and are described via
``annotateHandle`` operations in the instruction stream instead.

In LLVM we attempt to unify some of the alternative representations that are
present in DXC, with the aim of making handling of resources in the middle end
of the compiler simpler and more consistent.

Resource Type Information and Properties
========================================

There are a number of properties associated with a resource in DXIL.

`Resource ID`
   An arbitrary ID that must be unique per resource type (SRV, UAV, etc).

   In LLVM we don't bother representing this, instead opting to generate it at
   DXIL lowering time.

`Binding information`
   Information about where the resource comes from. This is either (a) a
   register space, lower bound in that space, and size of the binding, or (b)
   an index into a dynamic resource heap.

   In LLVM we represent binding information in the arguments of the
   :ref:`handle creation intrinsics <dxil-resources-handles>`. When generating
   DXIL we transform these calls to metadata, ``dx.op.createHandle``,
   ``dx.op.createHandleFromBinding``, ``dx.op.createHandleFromHeap``, and
   ``dx.op.createHandleForLib`` as needed.

`Type information`
   The type of data that's accessible via the resource. For buffers and
   textures this can be a simple type like ``float`` or ``float4``, a struct,
   or raw bytes. For constant buffers this is just a size. For samplers this is
   the kind of sampler.

   In LLVM we embed this information as a parameter on the ``target()`` type of
   the resource. See :ref:`dxil-resources-types-of-resource`.

`Resource kind information`
   The kind of resource. In HLSL we have things like ``ByteAddressBuffer``,
   ``RWTexture2D``, and ``RasterizerOrderedStructuredBuffer``. These map to a
   set of DXIL kinds like ``RawBuffer`` and ``Texture2D`` with fields for
   certain properties such as ``IsUAV`` and ``IsROV``.

   In LLVM we represent this in the ``target()`` type. We omit information
   that's deriveable from the type information, but we do have fields to encode
   ``IsWriteable``, ``IsROV``, and ``SampleCount`` when needed.

.. note:: TODO: There are two fields in the DXIL metadata that are not
   represented as part of the target type: ``IsGloballyCoherent`` and
   ``HasCounter``.

   Since these are derived from analysis, storing them on the type would mean
   we need to change the type during the compiler pipeline. That just isn't
   practical. It isn't entirely clear to me that we need to serialize this info
   into the IR during the compiler pipeline anyway - we can probably get away
   with an analysis pass that can calculate the information when we need it.

   If analysis is insufficient we'll need something akin to ``annotateHandle``
   (but limited to these two properties) or to encode these in the handle
   creation.

.. _dxil-resources-types-of-resource:

Types of Resource
=================

We define a set of ``TargetExtTypes`` that is similar to the HLSL
representations for the various resources, albeit with a few things
parameterized. This is different than DXIL, as simplifying the types to
something like "dx.srv" and "dx.uav" types would mean the operations on these
types would have to be overly generic.

Buffers
-------

.. code-block:: llvm

   target("dx.TypedBuffer", ElementType, IsWriteable, IsROV, IsSigned)
   target("dx.RawBuffer", ElementType, IsWriteable, IsROV)

We need two separate buffer types to account for the differences between the
16-byte `bufferLoad`_ / `bufferStore`_ operations that work on DXIL's
TypedBuffers and the `rawBufferLoad`_ / `rawBufferStore`_ operations that are
used for DXIL's RawBuffers and StructuredBuffers. We call the latter
"RawBuffer" to match the naming of the operations, but it can represent both
the Raw and Structured variants.

HLSL's Buffer and RWBuffer are represented as a TypedBuffer with an element
type that is a scalar integer or floating point type, or a vector of at most 4
such types. HLSL's ByteAddressBuffer is a RawBuffer with an `i8` element type.
HLSL's StructuredBuffers are RawBuffer with a struct, vector, or scalar type.

One unfortunate necessity here is that TypedBuffer needs an extra parameter to
differentiate signed vs unsigned ints. The is because in LLVM IR int types
don't have a sign, so to keep this information we need a side channel.

These types are generally used by BufferLoad and BufferStore operations, as
well as atomics.

There are a few fields to describe variants of all of these types:

.. list-table:: Buffer Fields
   :header-rows: 1

   * - Field
     - Description
   * - ElementType
     - Type for a single element, such as ``i8``, ``v4f32``, or a structure
       type.
   * - IsWriteable
     - Whether or not the field is writeable. This distinguishes SRVs (not
       writeable) and UAVs (writeable).
   * - IsROV
     - Whether the UAV is a rasterizer ordered view. Always ``0`` for SRVs.
   * - IsSigned
     - Whether an int element type is signed ("dx.TypedBuffer" only)

.. _bufferLoad: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#bufferload
.. _bufferStore: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#bufferstore
.. _rawBufferLoad: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#rawbufferload
.. _rawBufferStore: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#rawbufferstore

Resource Operations
===================

.. _dxil-resources-handles:

Resource Handles
----------------

We provide a few different ways to instantiate resources in the IR via the
``llvm.dx.handle.*`` intrinsics. These intrinsics are overloaded on return
type, returning an appropriate handle for the resource, and represent binding
information in the arguments to the intrinsic.

The three operations we need are ``llvm.dx.resource.handlefrombinding``,
``llvm.dx.handle.fromHeap``, and ``llvm.dx.handle.fromPointer``. These are
rougly equivalent to the DXIL operations ``dx.op.createHandleFromBinding``,
``dx.op.createHandleFromHeap``, and ``dx.op.createHandleForLib``, but they fold
the subsequent ``dx.op.annotateHandle`` operation in. Note that we don't have
an analogue for `dx.op.createHandle`_, since ``dx.op.createHandleFromBinding``
subsumes it.

We diverge from DXIL and index from the beginning of the binding rather than
indexing from the beginning of the binding space. This matches the semantics
more clearly and avoids a non-obvious invariant in what constitutes valid
arguments.

.. _dx.op.createHandle: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#resource-handles

.. list-table:: ``@llvm.dx.resource.handlefrombinding``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - A ``target()`` type
     - A handle which can be operated on
   * - ``%reg_space``
     - 1
     - ``i32``
     - Register space ID in the root signature for this resource.
   * - ``%lower_bound``
     - 2
     - ``i32``
     - Lower bound of the binding in its register space.
   * - ``%range_size``
     - 3
     - ``i32``
     - Range size of the binding.
   * - ``%index``
     - 4
     - ``i32``
     - Index from the beginning of the binding.
   * - ``%non-uniform``
     - 5
     - i1
     - Must be ``true`` if the resource index may be non-uniform.

.. note:: TODO: Can we drop the uniformity bit? I suspect we can derive it from
          uniformity analysis...

Examples:

.. code-block:: llvm

   ; RWBuffer<float4> Buf : register(u5, space3)
   %buf = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
        @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0(
            i32 3, i32 5, i32 1, i32 0, i1 false)

   ; RWBuffer<int> Buf : register(u7, space2)
   %buf = call target("dx.TypedBuffer", i32, 1, 0, 1)
        @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(
            i32 2, i32 7, i32 1, i32 0, i1 false)

   ; Buffer<uint4> Buf[24] : register(t3, space5)
   %buf = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
        @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_0_0t(
            i32 2, i32 7, i32 24, i32 0, i1 false)

   ; struct S { float4 a; uint4 b; };
   ; StructuredBuffer<S> Buf : register(t2, space4)
   %buf = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
       @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
           i32 4, i32 2, i32 1, i32 0, i1 false)

   ; ByteAddressBuffer Buf : register(t8, space1)
   %buf = call target("dx.RawBuffer", i8, 0, 0)
       @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(
           i32 1, i32 8, i32 1, i32 0, i1 false)

   ; RWBuffer<float4> Global[3] : register(u6, space5)
   ; RWBuffer<float4> Buf = Global[2];
   %buf = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0(
           i32 5, i32 6, i32 3, i32 2, i1 false)

.. list-table:: ``@llvm.dx.handle.fromHeap``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - A ``target()`` type
     - A handle which can be operated on
   * - ``%index``
     - 0
     - ``i32``
     - Index of the resource to access.
   * - ``%non-uniform``
     - 1
     - i1
     - Must be ``true`` if the resource index may be non-uniform.

Examples:

.. code-block:: llvm

   ; RWStructuredBuffer<float4> Buf = ResourceDescriptorHeap[2];
   declare
     target("dx.RawBuffer", <4 x float>, 1, 0)
     @llvm.dx.handle.fromHeap.tdx.RawBuffer_v4f32_1_0(
         i32 %index, i1 %non_uniform)
   ; ...
   %buf = call target("dx.RawBuffer", <4 x f32>, 1, 0)
               @llvm.dx.handle.fromHeap.tdx.RawBuffer_v4f32_1_0(
                   i32 2, i1 false)

Accessing Resources as Memory
-----------------------------

*relevant types: Buffers, CBuffer, and Textures*

Loading and storing from resources is generally represented in LLVM using
operations on memory that is only accessible via a handle object. Given a
handle, `llvm.dx.resource.getpointer` gives a pointer that can be used to read
and (depending on type) write to the resource.

Accesses using `llvm.dx.resource.getpointer` are replaced with direct load and
store operations in the `DXILResourceAccess` pass. These direct loads and
stores are described later in this document.

.. note:: Currently the pointers returned by `dx.resource.getpointer` are in
          the default address space, but that will likely change in the future.

.. list-table:: ``@llvm.dx.resource.getpointer``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - Pointer
     - A pointer to an object in the buffer
   * - ``%buffer``
     - 0
     - ``target(dx.TypedBuffer, ...)``
     - The buffer to access
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer

Examples:

.. code-block:: llvm

   %ptr = call ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_v4f32_0_0_0t(
       target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i32 %index)

Loads, Samples, and Gathers
---------------------------

*relevant types: Buffers, CBuffers, and Textures*

All load, sample, and gather operations in DXIL return a `ResRet`_ type, and
CBuffer loads return a similar `CBufRet`_ type. These types are structs
containing 4 elements of some basic type, and in the case of `ResRet` a 5th
element that is used by the `CheckAccessFullyMapped`_ operation. Some of these
operations, like `RawBufferLoad`_ include a mask and/or alignment that tell us
some information about how to interpret those four values.

In the LLVM IR representations of these operations we instead return scalars or
vectors, but we keep the requirement that we only return up to 4 elements of a
basic type. This avoids some unnecessary casting and structure manipulation in
the intermediate format while also keeping lowering to DXIL straightforward.

LLVM intrinsics that map to operations returning `ResRet` return an anonymous
struct with element-0 being the scalar or vector type, and element-1 being the
``i1`` result of a ``CheckAccessFullyMapped`` call. We don't have a separate
call to ``CheckAccessFullyMapped`` at all, since that's the only operation that
can possibly be done on this value. In practice this may mean we insert a DXIL
operation for the check when this was missing in the HLSL source, but this
actually matches DXC's behaviour in practice.

For TypedBuffer and Texture, we map directly from the contained type of the
resource to the return value of the intrinsic. Since these resources are
constrained to contain only scalars and vectors of up to 4 elements, the
lowering to DXIL ops is generally straightforward. The one exception we have
here is that `double` types in the elements are special - these are allowed in
the LLVM intrinsics, but are lowered to pairs of `i32` followed by
``MakeDouble`` operations for DXIL.

.. _ResRet: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#resource-operation-return-types
.. _CBufRet: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#cbufferloadlegacy
.. _CheckAccessFullyMapped: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/checkaccessfullymapped
.. _RawBufferLoad: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#rawbufferload

.. list-table:: ``@llvm.dx.resource.load.typedbuffer``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - A structure of the contained type and the check bit
     - The data loaded from the buffer and the check bit
   * - ``%buffer``
     - 0
     - ``target(dx.TypedBuffer, ...)``
     - The buffer to load from
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer

Examples:

.. code-block:: llvm

   %ret = call {<4 x float>, i1}
       @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_0_0_0t(
           target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i32 %index)
   %ret = call {float, i1}
       @llvm.dx.resource.load.typedbuffer.f32.tdx.TypedBuffer_f32_0_0_0t(
           target("dx.TypedBuffer", float, 0, 0, 0) %buffer, i32 %index)
   %ret = call {<4 x i32>, i1}
       @llvm.dx.resource.load.typedbuffer.v4i32.tdx.TypedBuffer_v4i32_0_0_0t(
           target("dx.TypedBuffer", <4 x i32>, 0, 0, 0) %buffer, i32 %index)
   %ret = call {<4 x half>, i1}
       @llvm.dx.resource.load.typedbuffer.v4f16.tdx.TypedBuffer_v4f16_0_0_0t(
           target("dx.TypedBuffer", <4 x half>, 0, 0, 0) %buffer, i32 %index)
   %ret = call {<2 x double>, i1}
       @llvm.dx.resource.load.typedbuffer.v2f64.tdx.TypedBuffer_v2f64_0_0t(
           target("dx.TypedBuffer", <2 x double>, 0, 0, 0) %buffer, i32 %index)

For RawBuffer, an HLSL load operation may return an arbitrarily sized result,
but we still constrain the LLVM intrinsic to return only up to 4 elements of a
basic type. This means that larger loads are represented as a series of loads,
which matches DXIL. Unlike in the `RawBufferLoad`_ operation, we do not need
arguments for the mask/type size and alignment, since we can calculate these
from the return type of the load during lowering.

.. _RawBufferLoad: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#rawbufferload

.. list-table:: ``@llvm.dx.resource.load.rawbuffer``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - A structure of a scalar or vector and the check bit
     - The data loaded from the buffer and the check bit
   * - ``%buffer``
     - 0
     - ``target(dx.RawBuffer, ...)``
     - The buffer to load from
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer
   * - ``%offset``
     - 2
     - ``i32``
     - Offset into the structure at the given index

Examples:

.. code-block:: llvm

   ; float
   %ret = call {float, i1}
       @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_0_0_0t(
           target("dx.RawBuffer", float, 0, 0, 0) %buffer,
           i32 %index,
           i32 0)
   %ret = call {float, i1}
       @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_i8_0_0_0t(
           target("dx.RawBuffer", i8, 0, 0, 0) %buffer,
           i32 %byte_offset,
           i32 0)

   ; float4
   %ret = call {<4 x float>, i1}
       @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_v4f32_0_0_0t(
           target("dx.RawBuffer", float, 0, 0, 0) %buffer,
           i32 %index,
           i32 0)
   %ret = call {float, i1}
       @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_i8_0_0_0t(
           target("dx.RawBuffer", i8, 0, 0, 0) %buffer,
           i32 %byte_offset,
           i32 0)

   ; struct S0 { float4 f; int4 i; };
   %ret = call {<4 x float>, i1}
       @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
           target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0, 0) %buffer,
           i32 %index,
           i32 0)
   %ret = call {<4 x i32>, i1}
       @llvm.dx.resource.load.rawbuffer.v4i32.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
           target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0, 0) %buffer,
           i32 %index,
           i32 1)

   ; struct Q { float4 f; int3 i; }
   ; struct R { int z; S x; }
   %ret = call {i32, i1}
       @llvm.dx.resource.load.rawbuffer.i32(
           target("dx.RawBuffer", {i32, {<4 x float>, <3 x i32>}}, 0, 0, 0)
               %buffer, i32 %index, i32 0)
   %ret = call {<4 x float>, i1}
       @llvm.dx.resource.load.rawbuffer.i32(
           target("dx.RawBuffer", {i32, {<4 x float>, <3 x i32>}}, 0, 0, 0)
               %buffer, i32 %index, i32 4)
   %ret = call {<3 x i32>, i1}
       @llvm.dx.resource.load.rawbuffer.i32(
           target("dx.RawBuffer", {i32, {<4 x float>, <3 x i32>}}, 0, 0, 0)
               %buffer, i32 %index, i32 20)

   ; byteaddressbuf.Load<int64_t4>
   %ret = call {<4 x i64>, i1}
       @llvm.dx.resource.load.rawbuffer.v4i64.tdx.RawBuffer_i8_0_0t(
           target("dx.RawBuffer", i8, 0, 0, 0) %buffer,
           i32 %byte_offset,
           i32 0)

Texture and Typed Buffer Stores
-------------------------------

*relevant types: Textures and TypedBuffer*

The `TextureStore`_ and `BufferStore`_ DXIL operations always write all four
32-bit components to a texture or a typed buffer. While both operations include
a mask parameter, it is specified that the mask must cover all components when
used with these types.

The store operations that we define as intrinsics behave similarly, and will
only accept writes to the whole of the contained type. This differs from the
loads above, but this makes sense to do from a semantics preserving point of
view. Thus, texture and buffer stores may only operate on 4-element vectors of
types that are 32-bits or fewer, such as ``<4 x i32>``, ``<4 x float>``, and
``<4 x half>``, and 2 element vectors of 64-bit types like ``<2 x double>`` and
``<2 x i64>``.

.. _BufferStore: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#bufferstore
.. _TextureStore: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#texturestore

Examples:

.. list-table:: ``@llvm.dx.resource.store.typedbuffer``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - ``void``
     -
   * - ``%buffer``
     - 0
     - ``target(dx.TypedBuffer, ...)``
     - The buffer to store into
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer
   * - ``%data``
     - 2
     - A 4- or 2-element vector of the type of the buffer
     - The data to store

Examples:

.. code-block:: llvm

   call void @llvm.dx.resource.store.typedbuffer.tdx.Buffer_v4f32_1_0_0t(
       target("dx.TypedBuffer", f32, 1, 0) %buf, i32 %index, <4 x f32> %data)
   call void @llvm.dx.resource.store.typedbuffer.tdx.Buffer_v4f16_1_0_0t(
       target("dx.TypedBuffer", f16, 1, 0) %buf, i32 %index, <4 x f16> %data)
   call void @llvm.dx.resource.store.typedbuffer.tdx.Buffer_v2f64_1_0_0t(
       target("dx.TypedBuffer", f64, 1, 0) %buf, i32 %index, <2 x f64> %data)
