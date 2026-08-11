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
