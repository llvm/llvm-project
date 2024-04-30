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

Samplers
--------

.. code-block:: llvm

   target("dx.Sampler", SamplerType)

The "dx.Sampler" type is used to represent sampler state. The sampler type is
an enum value from the DXIL ABI, and these appear in sampling operations as
well as LOD calculations and texture gather.

Constant Buffers
----------------

.. code-block:: llvm

   target("dx.CBuffer", BufferSize)

The "dx.CBuffer" type is a constant buffer of the given size. Note that despite
the name this is distinct from the buffer types, and can only be read using the
``llvm.dx.cbufferLoad`` operation.

Buffers
-------

.. code-block:: llvm

   target("dx.Buffer", ElementType, IsWriteable, IsROV)

There is only one buffer type. This can represent both UAVs and SRVs via the
``IsWriteable`` field. Since the type that's encoded is an llvm type, it
handles both ``Buffer`` and ``StructuredBuffer`` uniformly. For ``RawBuffer``,
the type is ``i8``, which is unambiguous since ``char`` isn't a legal type in
HLSL.

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

Textures
--------

.. code-block:: llvm

   target("dx.Texture1D", ElementType, IsWriteable, IsROV)
   target("dx.Texture1DArray", ...)
   target("dx.Texture2D", ...)
   target("dx.Texture2DArray", ...)
   target("dx.Texture3D", ...)
   target("dx.TextureCUBE", ...)
   target("dx.TextureCUBEArray", ...)

   target("dx.Texture2DMS", ElementType, IsWriteable, IsROV, SampleCount)
   target("dx.Texture2DMSArray", ...)

   target("dx.FeedbackTexture2D", ElementType, IsWriteable, IsROV, FeedbackType)
   target("dx.FeedbackTexture2DArray", ...)

There are a number of texture types, but they are mostly interestingly
different in their dimensions. These are distinct so that we can overload the
various sample and texture load/store operations such that their parameters are
appropriate to the type.

.. list-table:: Texture Fields
   :header-rows: 1

   * - Field
     - Description
   * - ElementType
     - Type for a single element, such as ``i8``, ``v4f32``, or a structure
       type.
   * - IsWriteable
     - Whether or not the field is writeable. This distinguishes SRVs (not
       writeable) and UAVs (writeable).
   * - SampleCount
     - Sample count for a multisampled texture.
   * - FeedbackType
     - Feedback type for a feedback texture.

Raytracing Resources
--------------------

.. code-block:: llvm

   target("dx.RTAccelerationStructure")

.. note:: TODO: Describe RTAccelerationStructure

Resource Operations
===================

.. _dxil-resources-handles:

Resource Handles
----------------

We provide a few different ways to instantiate resources in the IR via the
``llvm.dx.handle.*`` intrinsics. These intrinsics are overloaded on return
type, returning an appropriate handle for the resource, and represent binding
information in the arguments to the intrinsic.

The three operations we need are ``llvm.dx.handle.fromBinding``,
``llvm.dx.handle.fromHeap``, and ``llvm.dx.handle.fromPointer``. These are
rougly equivalent to the DXIL operations ``dx.op.createHandleFromBinding``,
``dx.op.createHandleFromHeap``, and ``dx.op.createHandleForLib``, but they fold
the subsequent ``dx.op.annotateHandle`` operation in. Note that we don't have
an analogue for `dx.op.createHandle`_, since ``dx.op.createHandleFromBinding``
subsumes it.

.. _dx.op.createHandle: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#resource-handles

.. list-table:: ``@llvm.dx.handle.fromBinding``
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
     - Index of the resource to access.
   * - ``%non-uniform``
     - 5
     - i1
     - Must be ``true`` if the resource index may be non-uniform.

.. note:: TODO: Can we drop the uniformity bit? I suspect we can derive it from
          uniformity analysis...

Examples:

.. code-block:: llvm

   ; RWBuffer<float4> Buf : register(u5, space3)
   %buf = call target("dx.Buffer", <4 x float>, 1, 0)
               @llvm.dx.handle.fromBinding.tdx.Buffer_v4f32_1_0(
                   i32 3, i32 5, i32 1, i32 0, i1 false)

   ; RWBuffer<uint> Buf : register(u7, space2)
   %buf = call target("dx.Buffer", i32, 1, 0)
               @llvm.dx.handle.fromBinding.tdx.Buffer_i32_1_0t(
                   i32 2, i32 7, i32 1, i32 0, i1 false)

   ; Buffer<uint4> Buf[24] : register(t3, space5)
   %buf = call target("dx.Buffer", <4 x i32>, 0, 0)
               @llvm.dx.handle.fromBinding.tdx.Buffer_v4i32_0_0t(
                   i32 2, i32 7, i32 24, i32 0, i1 false)

   ; struct S { float4 a; uint4 b; };
   ; StructuredBuffer<S> Buf : register(t2, space4)
   %buf = call target("dx.Buffer", {<4 x f32>, <4 x i32>}, 0, 0)
               @llvm.dx.handle.fromBinding.tdx.Buffer_sl_v4f32v4i32s_0_0t(
                   i32 4, i32 2, i32 1, i32 0, i1 false)

   ; ByteAddressBuffer Buf : register(t8, space1)
   %buf = call target("dx.Buffer", i8, 0, 0)
               @llvm.dx.handle.fromBinding.tdx.Buffer_i8_0_0t(
                   i32 1, i32 8, i32 1, i32 0, i1 false)

   ; cbuffer cb0 {
   ;   float4 g_MaxThreadIter : packoffset(c0);
   ;   float4 g_Window : packoffset(c1);
   ; }
   %cb0 = call target("dx.CBuffer", 32)
               @llvm.dx.handle.fromBinding.tdx.CBuffer_32t(
                   i32 0, i32 0, i32 1, i32 0, i1 false)

   ; Texture2D<float4> ColorMapTexture : register(t3);
   %tex = call target("dx.Texture2D", <4 x f32>, 0, 0)
               @llvm.dx.handle.fromBinding.tdx.Texture2D_v4f32_0_0t(
                   i32 0, i32 3, i32 1, i32 0, i1 false)

   ; Texture1D<float4> Buf[5] : register(t3);
   ; Texture1D<float4> B = Buf[NonUniformResourceIndex(i)];
   %tex = call target("dx.Texture1D", <4 x f32>, 0, 0)
               @llvm.dx.handle.fromBinding.tdx.Texture1D_v4f32_0_0t(
                   i32 0, i32 3, i32 5, i32 %i, i1 true)

   ; SamplerState ColorMapSampler : register(s0);
   %smp = call target("dx.Sampler", 0)
               @llvm.dx.handle.fromBinding.tdx.Sampler_0t(
                   i32 0, i32 0, i32 1, i32 0, i1 false)

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
     target("dx.Buffer", <4 x float>, 1, 0)
     @llvm.dx.handle.fromHeap.tdx.Buffer_v4f32_1_0(
         i32 %index, i1 %non_uniform)
   ; ...
   %buf = call target("dx.Buffer", <4 x f32>, 1, 0)
               @llvm.dx.handle.fromHeap.tdx.Buffer_v4f32_1_0(
                   i32 2, i1 false)

   ; struct S { float f; };
   ; ConstantBuffer<S> CB = ResourceDescriptorHeap[0];
   %cb0 = call target("dx.CBuffer", 4)
               @llvm.dx.handle.fromBinding.tdx.CBuffer_4t(
                   i32 0, i1 false)

   ; Texture2D<float4> ColorMapTexture : register(t3);
   %tex = call target("dx.Texture2D", <4 x f32>, 0, 0)
               @llvm.dx.handle.fromBinding.tdx.Texture2D_v4f32_0_0t(
                   i32 0, i1 false)

   ; Texture1D<float4> Buf[5] : register(t3);
   ; Texture1D<float4> B = Buf[NonUniformResourceIndex(i)];
   %tex = call target("dx.Texture1D", <4 x f32>, 0, 0)
               @llvm.dx.handle.fromHeap.tdx.Texture1D_v4f32_0_0t(
                   i32 %i, i1 true)

   ; SamplerState ColorMapSampler =  ResourceDescriptorHeap[3];
   %smp = call target("dx.Sampler", 0)
               @llvm.dx.handle.fromBinding.tdx.Sampler_0t(
                   i32 3, i1 false)

Buffer Loads and Stores
-----------------------

*relevant types: Buffers*

We separate loading from buffers into two operations, ``llvm.dx.bufferLoad``
and ``llvm.dx.bufferLoadComponent``. Store operations consist of their inverse,
``llvm.dx.bufferStore`` and ``llvm.dx.bufferStoreComponent``. These map to the
DXIL `rawBufferLoad`_ and `rawBufferStore`_ operations (and their older non-raw
counterparts).

We opt for two different intrinsics to best support the two main ways of
accessing buffer data.

The ``llvm.dx.bufferLoad`` intrinsic can return either a single element of a
buffer or a vector of consecutive elements. This makes accessing buffers of
scalars and simple vectors like `float4` simple, and is also convenient when
loading an entire `struct`. The variant with a vector of elements returned is
most useful for raw buffers, where we can load a number of bytes and `bitcast`
to the appropriate type, but can also be used to preserve the information that
we're loading data in bulk if needed.

The ``llvm.dx.bufferLoadComponent`` intrinsic has an extra index so that it can
be used to access specific struct elements or a particular component of a
simple vector type, like ``x`` of a ``float4``. This API gives the same
flexibility as the DXIL ``rawBufferLoad`` operation, but in a slightly more
readable way since it avoids ``undef`` values and bit masks, as well as using
an index instead of a byte offset.

The types involved in the store intrinsics match the load intrinsics.

When lowering these to the DXIL operations we need to pay attention to the DXIL
version and the type of data in the buffer. Post DXIL 1.2 structured and byte
address (or raw) buffers prefer RawBufferLoad and Store, whereas TypedBuffer
always uses BufferLoad and Store. For the raw operations, we need to provide an
alignment, but this can be derived from the buffer types in the LLVM
intrinsics.

.. note:: TODO: Can we always derive the alignment late, or do we need to
          parametrize these ops?

.. _rawBufferLoad: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#rawbufferload
.. _rawBufferStore: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#rawbufferstore

.. note:: TODO: We need to account for `CheckAccessFullyMapped`_ here.

   In DXIL the load operations always return an ``i32`` status value, but this
   isn't very ergonomic when it isn't used. We can (1) bite the bullet and have
   the loads return `{%ret_type, %i32}` all the time, (2) create a variant or
   update the signature iff the status is used, or (3) hide this in a sideband
   channel somewhere. I'm leaning towards (2), but could probably be convinced
   that the ugliness of (1) is worth the simplicity.

.. _CheckAccessFullyMapped: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/checkaccessfullymapped

.. list-table:: ``@llvm.dx.bufferLoad``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - The element type of the buffer, or a vector of the same.
     - The data loaded from the buffer
   * - ``%buffer``
     - 0
     - ``target(dx.Buffer, ...)``
     - The buffer to load from
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer

Examples:

.. code-block:: llvm

   ; Load from a buffer containing float4
   %ret = call <4 x float> @llvm.dx.bufferLoad.v4f32.tdx.Buffer_v4f32_0_0t(
       target("dx.Buffer", <4 x f32>, 0, 0) %buffer, i32 %index)
   ; Load a single element from a buffer containing float
   %ret = call float @llvm.dx.bufferLoad.f32.tdx.Buffer_f32_0_0t(
       target("dx.Buffer", f32, 0, 0) %buffer, i32 %index)
   ; Load 4 elements from a buffer containing float
   %ret = call <4 x float> @llvm.dx.bufferLoad.v4f32.tdx.Buffer_f32_0_0t(
       target("dx.Buffer", f32, 0, 0) %buffer, i32 %index)
   ; Load a struct
   %ret = call {f32, i32}
               @llvm.dx.bufferLoad.sl_f32i32s.tdx.Buffer_sl_f32i32s_0_0t(
                   target("dx.Buffer", {f32, i32}, 0, 0) %buffer, i32 %index)

   ; Load an i32 from a byte address buffer
   %ret = call <4 x i8> @llvm.dx.bufferLoad.v4i8.tdx.Buffer_i8_0_0t(
       target("dx.Buffer", i8, 0, 0) %buffer, i32 %index)
   %cast = bitcast <4 x i8> %ret to i32

.. list-table:: ``@llvm.dx.bufferLoadComponent``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - The type of the component
     - The data loaded from the buffer
   * - ``%buffer``
     - 0
     - ``target(dx.Buffer, ...)``
     - The buffer to load from
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer
   * - ``%component``
     - 2
     - ``i32``
     - Index of the component to access. Must be constant for buffers
       containing structs.

Examples:

.. code-block:: llvm

   ; Load the `y` component from a buffer containing float4
   %ret = call float @llvm.dx.bufferLoadComponent.f32.tdx.Buffer_v4f32_0_0t(
       target("dx.Buffer", <4 x f32>, 0, 0) %buffer, i32 %index, i32 1)
   ; Load the double from a struct containing an int, a float, and a double
   %ret = call f64 @llvm.dx.bufferLoad.f64.tdx.Buffer_sl_i32f32f64s_0_0t(
       target("dx.Buffer", {i32, f32, f64}, 0, 0) %buffer, i32 %index, i32 2)

.. list-table:: ``@llvm.dx.bufferStore``
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
     - ``target(dx.Buffer, ...)``
     - The buffer to store into
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer
   * - ``%data``
     - 2
     - The element type of the buffer, or a vector of the same.
     - The data to store

Examples:

.. code-block:: llvm

   call void @llvm.dx.bufferStore.tdx.Buffer_v4f32_1_0t.v4f32(
       target("dx.Buffer", <4 x f32>, 1, 0) %buf, i32 %index, <4 x f32> %data)
   call void @llvm.dx.bufferStore.tdx.Buffer_f32_1_0t.v4f32(
       target("dx.Buffer", f32, 1, 0) %buf, i32 %index, <4 x f32> %data)
   call void @llvm.dx.bufferStore.tdx.Buffer_f32_1_0t.f32(
       target("dx.Buffer", f32, 1, 0) %buf, i32 %index, f32 %data)

   %vec = bitcast f32 %data to <4 x i8>
   call void @llvm.dx.bufferStore.tdx.Buffer_i8_1_0t.v4i8(
       target("dx.Buffer", i8, 1, 0) %buf, i32 %index, <4 x i8> %vec)

.. list-table:: ``@llvm.dx.bufferStoreComponent``
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
     - ``target(dx.Buffer, ...)``
     - The buffer to store into
   * - ``%index``
     - 1
     - ``i32``
     - Index into the buffer
   * - ``%component``
     - 2
     - ``i32``
     - Index of the component to store. Must be constant for buffers containing
       structs.
   * - ``%data``
     - 3
     - The element type of the buffer, or a vector of the same.
     - The data to store

Examples:

.. code-block:: llvm

   ; Store the `y` component from a buffer containing float4
   call void @llvm.dx.bufferStoreComponent.tdx.Buffer_v4f32_0_0t.f32(
       target("dx.Buffer", <4 x f32>, 0, 0) %buf, i32 %index, i32 1, f32 %data)
   ; Store the float from a struct containing an int and a float
   call void @llvm.dx.bufferLoad.tdx.Buffer_sl_i32f32f64s_0_0t.f32(
       target("dx.Buffer", {i32, f32}, 0, 0) %buf, i32 %index, i32 1, f32 %data)

Constant Buffer Loads
---------------------

*relevant types: CBuffer*

Loading from constant buffers is relatively straightforward, and we model it
with a single intrinsic, `llvm.dx.cbufferLoad`. Similarly to how we handle
``bufferLoad``, we allow the intrinsic to load a scalar or a vector of
consecutive values as needed.

In DXIL there are two operations that load from a constant buffer -
`cBufferLoad`_ and ` and `cbufferLoadLegacy`_. These names can be confusing
because only ``cbufferLoadLegacy`` is widely supported by drivers. The "legacy"
op loads 16 bytes of data from the buffer, which is generally 4 32-bit values
or 2 64-bit values, whereas the largely unsupported replacement loads a scalar
directly. These differences are minor and we can easily lower to either from
our own generic intrinsic.

.. _cbufferLoadLegacy: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#cbufferloadlegacy
.. _cbufferLoad: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#cbufferload

.. list-table:: ``@llvm.dx.cbufferLoad``
   :header-rows: 1

   * - Argument
     -
     - Type
     - Description
   * - Return value
     -
     - The scalar or vector type to load.
     - Data loaded from the cbuffer.
   * - ``%cbuffer``
     - 0
     - ``target(dx.CBuffer, ...)``
     - The constant buffer to load from
   * - ``%offset``
     - 1
     - ``i32``
     - Byte offset to load from

Examples:

.. code-block:: llvm

   %ret = call float @llvm.dx.cbufferLoad.f32.tdx.CBuffer_16t(
       target("dx.CBuffer", 16) %cbuffer, i32 2)
   %ret = call <4 x float> @llvm.dx.cbufferLoad.v4f32.tdx.CBuffer_64t(
       target("dx.CBuffer", 16) %cbuffer, i32 0)

.. note:: TODO: Again, do we need alignment or can we rely on types here?

Texture Load and Store
----------------------

*relevant types: Textures*

.. note:: TODO: The texture load and store operations in DXIL have *a lot* of
          parameters, and they use an awkward trick where which ones are valid
          and which *must* be ``undef`` depends on the texture type. I'd rather
          not copy that approach.

          We have a few options:

          1. Encode the dimensions into the operation in a way similar to the
             approach described below in the ``GetDimensions`` section. I
             showed that approach there because here is a bit less tractable.
          2. Encode the dimensions as types on the ``dx.Texture`` target type.
             This gives us uniform structure of the operations at the expense
             of them being more awkward to work with.

GetDimensions
-------------

*relevant types: All Texture and Buffer types*

.. note:: TODO: This approach needs discussion. See the commentary in texture
          loads and stores above.

Looking up a resource's dimensions is done via a family of ``GetDimensions``
intrinsics. The equivalent `DXIL operation <GetDimensions_>`_ returns ``{ i32,
i32, i32, 32 }`` regardless of which dimensions are relevant for the resource,
which can be confusing. Instead, we define variants of the form
``GetDimensions.xyzsm`` where the returned elements are represented by an
initial:

- ``x`` is width
- ``y`` is height
- ``z`` is array size for 1D or 2D arrays, or depth for 3D textures
- ``s`` is number of samples in a multisampled texture
- ``m`` is the number of MIP levels in a mipmapped texture

The set of variants is not dynamic - these initials are always specified in
this order and there are exactly 10 valid configurations.

.. _GetDimensions: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#getdimensions

.. list-table:: ``GetDimensions`` variants
   :header-rows: 1

   * - Variant
     - Returns
     - Args
     - Resource types
   * - ``GetDimensions.x``
     - ``i32``
     - n/a
     - ``Buffer``, read-only ``Texture1D``
   * - ``GetDimensions.xm``
     - ``{i32, i32}``
     - ``%miplevel``
     - writeable ``Texture1D``
   * - ``GetDimensions.xy``
     - ``{i32, i32}``
     - n/a
     - read-only ``Texture2D`` and ``TextureCube``
   * - ``GetDimensions.xym``
     - ``{i32, i32, i32}``
     - ``%miplevel``
     - writeable ``Texture2D`` and ``TextureCube``
   * - ``GetDimensions.xyz``
     - ``{i32, i32, i32}``
     - n/a
     - readonly ``Texture2DArray``, ``Texture3D``, ``TextureCubeArray``
   * - ``GetDimensions.xyzm``
     - ``{i32, i32, i32, i32}``
     - ``%miplevel``
     - writeable ``Texture2DArray``, ``Texture3D``, ``TextureCubeArray``
   * - ``GetDimensions.xys``
     - ``{i32, i32, i32}``
     - n/a
     - ``Texture2DMS``
   * - ``GetDimensions.xyzs``
     - ``{i32, i32, i32, i32}``
     - n/a
     - ``Texture2DMSArray``

Examples:

.. code-block:: llvm

   declare i32
       @llvm.dx.GetDimensions.x.tdx.Buffer_f32_0_0t()
   declare i32
       @llvm.dx.GetDimensions.x.tdx.Texture1D_f32_0_0t()
   declare {i32, i32}
       @llvm.dx.GetDimensions.xm.tdx.Texture1D_f32_1_0t(i32 %miplevel)
   declare {i32, i32}
       @llvm.dx.GetDimensions.xy.tdx.Texture2D_f32_0_0t()
   declare {i32, i32}
       @llvm.dx.GetDimensions.xz.tdx.Texture1DArray_f32_0_0t()
   declare {i32, i32, i32, i32}
       @llvm.dx.GetDimensions.xyzm.tdx.Texture3D_f32_1_0t(i32 %miplevel)
   declare {i32, i32, i32, i32}
       @llvm.dx.GetDimensions.xyzs.tdx.Texture2DMSArray_f32_1_0_1t()

Samples
-------

*relevant types: Textures and Samples*

.. note:: TODO: Describe sampling.

   The following DXIL operations are relevant:
   - ``Sample``
   - ``SampleBias``
   - ``SampleLevel``
   - ``SampleGrad``
   - ``SampleCmp``
   - ``SampleCmpLevelZero``

CalculateLOD
------------

*relevant types: Textures and Samples*

.. note:: TODO: Describe ``CalculateLOD``.

TextureGather
-------------

*relevant types: Textures and Samples*

.. note:: TODO: Describe ``TextureGather`` and ``TextureGatherCmp``.

Texture2DMSGetSamplePosition
----------------------------

*relevant types: Texture2DMS and maybe Texture2DMSArray*

.. note:: TODO: Describe ``Texture2DMSGetSamplePosition``

Buffer Update Counter
---------------------

*relevant types: RWStructuredBuffer?*

.. note:: TODO: Describe ``BufferUpdateCounter``

   Note that we need to account for calculating HasCounter, as was mentioned in
   an earlier TODO.

Atomics
-------

*relevant types: Textures and Buffers*

.. note:: TODO: Describe ``AtomicBinOp`` and ``AtomicCompareExchange``

Reflection Information
======================

.. note:: TODO: Add some notes on the HLSL variable name and its relationship
          to reflection.

          Ideally this would boil down to a link to a separate "reflection"
          document.
