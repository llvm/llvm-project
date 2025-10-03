===============
Root Signatures
===============

.. contents::
   :local:

.. toctree::
   :hidden:

Overview
========

A root signature is used to describe what resources a shader needs access to
and how they're organized and bound in the pipeline. The DirectX Container
(DXContainer) contains a root signature part (RTS0), which stores this
information in a binary format. To assist with the construction of, and
interaction with, a root signature is represented as metadata
(``dx.rootsignatures`` ) in the LLVM IR. The metadata can then be converted to
its binary form, as defined in
`llvm/include/llvm/llvm/Frontend/HLSL/RootSignatureMetadata.h
<https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Frontend/HLSL/RootSignatureMetadata.h>`_.
This document serves as a reference for the metadata representation of a root
signature for users to interface with.

Metadata Representation
=======================

Consider the reference root signature, then the following sections describe the
metadata representation of this root signature and the corresponding operands.

.. code-block:: HLSL

  RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT),
  RootConstants(b0, space = 1, num32Constants = 3),
  CBV(b1, flags = 0),
  StaticSampler(
    filter = FILTER_MIN_MAG_POINT_MIP_LINEAR,
    addressU = TEXTURE_ADDRESS_BORDER,
  ),
  DescriptorTable(
    visibility = VISIBILITY_ALL,
    SRV(t0, flags = DATA_STATIC_WHILE_SET_AT_EXECUTE),
    UAV(
      numDescriptors = 5, u1, space = 10, offset = 5,
      flags = DATA_VOLATILE
    )
  )

.. note::

  A root signature does not necessarily have a unique metadata representation.
  Futher, a malformed root signature can be represented in the metadata format,
  (eg. mixing Sampler and non-Sampler descriptor ranges), and so it is the
  user's responsibility to verify that it is a well-formed root signature.

Named Root Signature Table
==========================

.. code-block:: LLVM

  !dx.rootsignatures = !{!0}

A named metadata node, ``dx.rootsignatures``` is used to identify the root
signature table. The table itself is a list of references to function/root
signature pairs.

Function/Root Signature Pair
============================

.. code-block:: LLVM

  !1 = !{ptr @main, !2, i32 2 }

The function/root signature associates a function (the first operand) with a
reference to a root signature (the second operand). The root signature version
(the third operand) used for validation logic and binary format follows.

Root Signature
==============

.. code-block:: LLVM

  !2 = !{ !3, !4, !5, !6, !7 }

The root signature itself simply consists of a list of references to its root
signature elements.

Root Signature Element
======================

A root signature element is identified by the first operand, which is a string.
The following root signature elements are defined:

================= ======================
Identifier String Root Signature Element
================= ======================
"RootFlags"       Root Flags
"RootConstants"   Root Constants
"RootCBV"         Root Descriptor
"RootSRV"         Root Descriptor
"RootUAV"         Root Descriptor
"StaticSampler"   Static Sampler
"DescriptorTable" Descriptor Table
================= ======================

Below is listed the representation for each type of root signature element.

Root Flags
==========

.. code-block:: LLVM

  !3 = { !"RootFlags", i32 1 }

======================= ====
Description             Type
======================= ====
`Root Signature Flags`_ i32
======================= ====

.. _Root Signature Flags: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_root_signature_flags

Root Constants
==============

.. code-block:: LLVM

  !4 = { !"RootConstants", i32 0, i32 1, i32 2, i32 3 }

==================== ====
Description          Type
==================== ====
`Shader Visibility`_ i32
Shader Register      i32
Register Space       i32
Number 32-bit Values i32
==================== ====

.. _Shader Visibility: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_shader_visibility

Root Descriptor
===============

As noted in the table above, the first operand will denote the type of
root descriptor.

.. code-block:: LLVM

  !5 = { !"RootCBV", i32 0, i32 1, i32 0, i32 0 }

======================== ====
Description              Type
======================== ====
`Shader Visibility`_     i32
Shader Register          i32
Register Space           i32
`Root Descriptor Flags`_ i32
======================== ====

.. _Root Descriptor Flags: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_root_descriptor_flags

Static Sampler
==============

.. code-block:: LLVM

  !6 = !{ !"StaticSampler", i32 1, i32 4, ... }; remaining operands omitted for space

==================== =====
Description          Type
==================== =====
`Filter`_            i32
`AddressU`_          i32
`AddressV`_          i32
`AddressW`_          i32
MipLODBias           float
MaxAnisotropy        i32
`ComparisonFunc`_    i32
`BorderColor`_       i32
MinLOD               float
MaxLOD               float
ShaderRegister       i32
RegisterSpace        i32
`Shader Visibility`_ i32
==================== =====

.. _Filter: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_filter
.. _AddressU: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_texture_address_mode
.. _AddressV: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_texture_address_mode
.. _AddressW: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_texture_address_mode
.. _ComparisonFunc: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_comparison_func>
.. _BorderColor: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_static_border_color>

Descriptor Table
================

A descriptor table consists of a visibility and the remaining operands are a
list of references to its descriptor ranges.

.. note::

  The term Descriptor Table Clause is synonymous with Descriptor Range when
  referencing the implementation details.

.. code-block:: LLVM

  !7 = { !"DescriptorTable", i32 0, !8, !9 }

========================= ================
Description               Type
========================= ================
`Shader Visibility`_      i32
Descriptor Range Elements Descriptor Range
========================= ================


Descriptor Range
================

Similar to a root descriptor, the first operand will denote the type of
descriptor range. It is one of the following types:

- "CBV"
- "SRV"
- "UAV"
- "Sampler"

.. code-block:: LLVM

  !8 = !{ !"SRV", i32 1, i32 0, i32 0, i32 -1, i32 4 }
  !9 = !{ !"UAV", i32 5, i32 1, i32 10, i32 5, i32 2 }

============================== ====
Description                    Type
============================== ====
Number of Descriptors in Range i32
Shader Register                i32
Register Space                 i32
`Offset`_                      i32
`Descriptor Range Flags`_      i32
============================== ====

.. _Offset: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_descriptor_range
.. _Descriptor Range Flags: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_descriptor_range_flags
