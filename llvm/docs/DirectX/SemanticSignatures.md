# Semantic Signatures

## Overview

A semantic signature describes the inputs and outputs of an HLSL shader entry
point: the semantics each value carries, its component type, and where it is
placed in the input/output register space. The DirectX Container (DXContainer)
stores this information in binary signature parts (`ISG1`, `OSG1`) and in the
pipeline state validation part (`PSV0`). To assist with the construction of, and
interaction with, these parts, a semantic signature is represented as metadata
(`dx.semantic.signatures`) in the LLVM IR. The metadata can then be converted to
its binary form, as defined in [SemanticSignatures.h]. This document serves as a
reference for the metadata representation of a semantic signature for users to
interface with.

[SemanticSignatures.h]: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Frontend/HLSL/SemanticSignatures.h

## Metadata Representation

Consider the reference shaders below, then the following sections describe the
metadata representation of their signatures and the corresponding operands.

```hlsl
float4 vs_main(float4 pos   : POSITION,
               float4 uv[2] : TEXCOORD0) : SV_Position {
  return pos + uv[0] + uv[1];
}

struct PSOut {
  float4 color : SV_Target0;
  float4 extra : SV_Target1;
};

PSOut ps_main(float4 pos : SV_Position,
              float4 uv0 : TEXCOORD0,
              float4 uv1 : TEXCOORD1) {
  PSOut o;
  o.color = pos + uv0;
  o.extra = float4(uv1.xyz, 1);
  return o;
}
```

> **Note:** A signature does not necessarily have a unique metadata
> representation. Further, a malformed signature can be represented in the
> metadata format, and so it is the user's responsibility to verify that it is a
> well-formed signature.

## Named Signature Table

```LLVM
!dx.semantic.signatures = !{!1, !2}
```

A named metadata node, `dx.semantic.signatures`, is used to identify the table
of per-entry-point semantic signatures. The table itself is a list of references
to function/signature triples. If no entry point has a signature, the named
metadata node may be omitted entirely.

## Function/Signature Triple

```LLVM
!1 = !{ ptr @vs_main, !3, !4 }
```

The function/signature triple associates an entry-point function (the first
operand) with its input signature element list (the second operand) and output
signature element list (the third operand). Either list may be `null`. An entry
function may appear at most once.

## Signature Element List

```LLVM
!3 = !{ !5, !6 }
```

A signature element list consists of a list of references to signature element
nodes.

## Signature Element

```LLVM
!5 = !{ i32 0, !"TEXCOORD", i32 9, i32 0, !50, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0 }
```

A signature element describes a single packed range of signature rows. It
retains all information needed to serialize into `ISG1`, `OSG1` and `PSV0`.

| Name                   | Type            | Description                                                                                                                                                                                                                        |
|------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Signature ID           | i32             | dense 0-based index within the entry function signature list; matches the operand of `llvm.dx.load.input` / `llvm.dx.store.output`                                                                                                 |
| Semantic Name          | metadata string | the semantic name (e.g. `!"TEXCOORD"`, `!"SV_Position"`)                                                                                                                                                                           |
| Component Type         | i32             | component type; see [`llvm::dxil::ElementType`][ElementType].                                                                                                                                                                     |
| Semantic Kind          | i32             | semantic kind; `Arbitrary` (0) for user-defined semantics, the corresponding `SV_*` value otherwise. See [`SEMANTIC_KIND`][DXContainerConstants]                                                                                 |
| Semantic Indices       | metadata node   | reference to a [semantic indices](#semantic-indices) node                                                                                                                                                                          |
| Interpolation Mode     | i32             | interpolation mode; see [`INTERPOLATION_MODE`][DXContainerConstants]                                                                                                                                                             |
| Rows                   | i32             | number of consecutive register rows occupied                                                                                                                                                                                       |
| Cols                   | i8              | number of components per row (1–4)                                                                                                                                                                                                 |
| Start Row              | i32             | starting register row; `-1` (`0xFFFFFFFF`) if unallocated                                                                                                                                                                          |
| Start Column           | i8              | starting component column; `-1` (`0xFF`) if unallocated, otherwise 0–3                                                                                                                                                             |
| Usage Mask             | i8              | 4-bit bitmask of components that are always read (input) or may be written (output).                                                                                                                                               |
| Dynamic Index Mask     | i8              | 4-bit bitmask of components that are dynamically indexed                                                                                                                                                                           |
| GS Output Stream Index | i32             | GS output stream index; 0 for non-GS stages                                                                                                                                                                                        |

[ElementType]: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/DXILABI.h
[DXContainerConstants]: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/BinaryFormat/DXContainerConstants.def

### Derived Container Fields

The following container fields are derived from the operands above:

- **Allocated**: allocated iff `StartRow != -1` and `StartCol != -1` (the
  sentinels are always set together).
- **DeclaredMask**: `((1 << Cols) - 1) << StartCol`.
- **AlwaysReads / NeverWrites**: `UsageMask` is written to `AlwaysReads` for
  inputs; for outputs `NeverWrites = ~UsageMask & DeclaredMask`.
- **MinPrecision**: from `CompType` plus the `UseMinPrecision` module flag.

## Semantic Indices

```LLVM
!50 = !{ i32 0 }
!51 = !{ i32 0, i32 1 }
```

A metadata node of one or more semantic indices. Its length must equal the
`Rows` field of the containing signature element.

## Signature Packing

Before a semantic signature is serialized, each element that participates in
packing is assigned a location in a fixed register space of 32 rows and 4
columns. An element occupies a rectangle of `Rows` consecutive registers and
`Cols` consecutive components. Its allocated location is recorded in
`StartRow` and `StartCol`.

The packing helper classifies each element from its semantic kind, shader stage,
and I/O type. Elements with the `NotAllocated` interpretation are accessed by
other means and retain the unallocated row and column sentinels. The remaining
interpretations accepted by a packing algorithm are assigned locations
according to that algorithm's rules. If an eligible element cannot be placed,
packing returns a `SignaturePackingError` identifying the element that failed.

The packing APIs and their in-memory element representation are declared in
[SemanticSignaturePacking.h].

[SemanticSignaturePacking.h]: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Frontend/HLSL/SemanticSignaturePacking.h

### Stacked Packing

Stacked packing is used for a vertex shader input signature. Eligible elements
are visited in declaration order. Each starts at column zero of the first row
after the preceding element, and a multi-row element occupies consecutive rows.
Elements are never co-packed into the unused columns of another element, and
interpolation mode, component type, and semantic interpretation do not otherwise
affect placement.

For example:

```hlsl
struct VSIn {
  float A       : A;
  float3 B[2]   : B;
  uint VertexID : SV_VertexID;
};
```

The signature is allocated as:

```text
reg0: A.x        | unused.yzw
reg1: B[0].xyz   | unused.w
reg2: B[1].xyz   | unused.w
reg3: VertexID.x | unused.yzw
```

### Indexed Packing

Indexed packing is used for a pixel shader output signature. Each eligible
`SV_Target` element occupies one row and starts at column zero. Its semantic
index directly selects that row, so declaration order does not affect placement
and rows without a corresponding semantic index remain unused. Elements that do
not contribute to the target register space remain unallocated.

For example:

```hlsl
struct PSOut {
  float4 Color3 : SV_Target3;
  float Color0  : SV_Target0;
  float2 Color2 : SV_Target2;
};
```

The signature is allocated as:

```text
reg0: Color0.x  | unused.yzw
reg1: unused.xyzw
reg2: Color2.xy | unused.zw
reg3: Color3.xyzw
```

### Prefix-Stable Packing

Prefix-stable packing is used for signatures that connect programmable shader
stages or carry patch constant data. Elements are visited in declaration order
and placed at the first compatible location in the 32-row by 4-column register
space. Once an element is placed it is never moved, so appending elements to a
signature does not change the locations assigned to its existing prefix.

Elements can share unused components in a row when all applicable packing
constraints are satisfied:

- Every element in a row must have a compatible interpolation mode.
- When native 16-bit types are enabled, every element in a row must have the
  same component width. Without native 16-bit types, min-precision values
  occupy 32-bit components.
- Components are ordered from arbitrary values, to system values, to system
  generated values.
- A system value or system generated value cannot be placed in a dynamically
  indexed row. Multi-row elements define the dynamically indexed range that
  they cover.

Some semantic interpretations require additional handling:

- `SV_ClipDistance` and `SV_CullDistance` are packed only with each other in
  dedicated rows. Together they may occupy at most eight components across at
  most two rows. The rows must be adjacent when a clip or cull element spans
  multiple rows.
- A multi-row tessellation factor is searched for only in the last column.
- Geometry shader output streams are packed independently.

For example:

```hlsl
struct VSOut {
  float3 A[3] : A;
  float1x2 B  : B;
  float2 C    : C;
  float D     : D;
};
```

Assuming `B` has column-major matrix orientation, the signature is allocated
as:

```text
reg0: A[0].xyz | B[0][0].w
reg1: A[1].xyz | B[0][1].w
reg2: A[2].xyz | D.w
reg3: C.xy     | unused.zw
```

### Optimized Packing

Optimized packing partitions eligible elements into groups and packs the groups
in this order:

1. Four-column arbitrary and system-value elements.
2. Multi-row tessellation factors, which are restricted to the last column.
3. Arbitrary elements.
4. System-value elements, including single-row tessellation factors.
5. `SV_ClipDistance` and `SV_CullDistance` elements.
6. System-generated-value elements.

Within each group, elements are ordered first by the numeric value of their
interpolation mode, then by decreasing row count, then by decreasing column
count, and finally by increasing signature ID. Component bit width is not a
sort key; it remains a compatibility constraint when elements are placed in a
row.

The sorted elements are then placed using the prefix-stable packing algorithm.
This is a greedy optimized ordering rather than an exhaustive search for a
minimum-row layout. Reordering can reduce the number of rows occupied, but
means that appending an element may change locations assigned to existing
elements.
