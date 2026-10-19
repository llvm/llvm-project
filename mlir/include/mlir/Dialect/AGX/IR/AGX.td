//===-- AGX.td - AGX dialect definitions -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AGX
#define AGX

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AGX dialect definition
//===----------------------------------------------------------------------===//

def AGX_Dialect : Dialect {
  let name = "agx";
  let cppNamespace = "::mlir::agx";
  let description = [{
    The `agx` dialect provides wrappers around the functionality of Apple
    GPUs (the AGX architecture) in the style of the `amdgpu` dialect. It is
    used together with generic dialects such as `gpu` and `vector`, and lowers
    to the `agxir` dialect, whose operations are 1:1 wrappers of the LLVM AGX
    backend's intrinsics.

    Operations belong here when they make AGX functionality work better with
    MLIR types or hide magic constants of the intrinsics:

    - `agx.shuffle` exchanges values of any type, splitting those wider than
      32 bits, with one operation for the SIMD-group and quad variants of
      every shuffle mode.
    - `agx.reduce` names the reduction instead of the intrinsic's operation
      number, and extends 16-bit values.
    - `agx.threadgroup_barrier` is an execution barrier that also orders
      threadgroup memory, like `amdgpu.lds_barrier`.
  }];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Enums and attributes
//===----------------------------------------------------------------------===//

def AGX_Scope : I32EnumAttr<"Scope", "The threads an operation spans", [
    I32EnumAttrCase<"simdgroup", 0>,
    I32EnumAttrCase<"quad", 1>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::agx";
}
def AGX_ScopeAttr : EnumAttr<AGX_Dialect, AGX_Scope, "scope"> {
  let assemblyFormat = "$value";
}

def AGX_MemoryScope : I32EnumAttr<"MemoryScope",
    "The threads a memory barrier orders accesses for", [
    I32EnumAttrCase<"threadgroup", 0>,
    I32EnumAttrCase<"device", 1>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::agx";
}
def AGX_MemoryScopeAttr : EnumAttr<AGX_Dialect, AGX_MemoryScope,
                                   "memory_scope"> {
  let assemblyFormat = "$value";
}

def AGX_ShuffleMode : I32EnumAttr<"ShuffleMode", "Lane selection of a shuffle",
  [
    I32EnumAttrCase<"idx", 0>,
    I32EnumAttrCase<"xor_", 1, "xor">,
    I32EnumAttrCase<"up", 2>,
    I32EnumAttrCase<"down", 3>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::agx";
}
def AGX_ShuffleModeAttr : EnumAttr<AGX_Dialect, AGX_ShuffleMode, "shuffle_mode"> {
  let assemblyFormat = "$value";
}

def AGX_ReduceKind : I32EnumAttr<"ReduceKind", "The operation of a reduction",
  [
    I32EnumAttrCase<"add", 0>,
    I32EnumAttrCase<"mul", 1>,
    I32EnumAttrCase<"minsi", 2>,
    I32EnumAttrCase<"maxsi", 3>,
    I32EnumAttrCase<"minui", 4>,
    I32EnumAttrCase<"maxui", 5>,
    I32EnumAttrCase<"minnumf", 6>,
    I32EnumAttrCase<"maxnumf", 7>,
    I32EnumAttrCase<"and_", 8, "and">,
    I32EnumAttrCase<"or_", 9, "or">,
    I32EnumAttrCase<"xor_", 10, "xor">
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::agx";
}
def AGX_ReduceKindAttr : EnumAttr<AGX_Dialect, AGX_ReduceKind, "reduce_kind"> {
  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class AGX_Op<string mnemonic, list<Trait> traits = []>
    : Op<AGX_Dialect, mnemonic, traits>;

def AGX_ThreadgroupBarrierOp : AGX_Op<"threadgroup_barrier"> {
  let summary = "Barrier that also orders threadgroup memory";
  let description = [{
    `agx.threadgroup_barrier` waits for all threads of the threadgroup and
    makes the threadgroup memory writes before it visible to the reads after
    it, like `__syncthreads()`.

    ```mlir
    agx.threadgroup_barrier
    ```
  }];
  let assemblyFormat = "attr-dict";
}

def AGX_ShuffleOp : AGX_Op<"shuffle", [AllTypesMatch<["value", "result"]>]> {
  let summary = "Read a value from another thread";
  let description = [{
    Reads `value` from another thread of the SIMD-group (the default)
    or of the quad (`scope quad`). The lane is
    `index` itself (`idx`), the current lane xor `index` (`xor`), or the
    current lane minus or plus `index` (`up`, `down`).

    Values of 16 or 32 bits are shuffled directly; wider values and vectors
    are split into 32-bit parts.

    ```mlir
    %0 = agx.shuffle xor %v, %c1 : f32
    %1 = agx.shuffle down %w, %c4 scope quad : vector<2xf32>
    ```
  }];
  let arguments = (ins AGX_ShuffleModeAttr:$mode,
                       AnyTypeOf<[AnyInteger, AnyFloat, AnyVectorOfNonZeroRank]>:$value,
                       I32:$index,
                       DefaultValuedAttr<AGX_ScopeAttr,
                                         "::mlir::agx::Scope::simdgroup">:$scope);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $mode $value `,` $index (`scope` $scope^)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AGX_ReduceOp : AGX_Op<"reduce", [AllTypesMatch<["value", "result"]>]> {
  let summary = "Reduction across the active threads";
  let description = [{
    Combines `value` across the active threads of the SIMD-group or of the
    quad with the operation `kind`. With `prefix`, each thread receives the
    combination of the threads before it instead.

    Integer reductions take 16 or 32-bit integers (16-bit values are
    extended according to the signedness of the operation); floating point
    reductions take `f16` or `f32`.

    ```mlir
    %sum = agx.reduce add %v : f32
    %max = agx.reduce maxui %i scope quad : i32
    %pre = agx.reduce add %v prefix : i32
    ```
  }];
  let arguments = (ins AGX_ReduceKindAttr:$kind,
                       AnyTypeOf<[I16, I32, F16, F32]>:$value,
                       DefaultValuedAttr<AGX_ScopeAttr,
                                         "::mlir::agx::Scope::simdgroup">:$scope,
                       UnitAttr:$prefix);
  let results = (outs AnyTypeOf<[I16, I32, F16, F32]>:$result);
  let assemblyFormat = [{
    $kind $value (`prefix` $prefix^)? (`scope` $scope^)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AGX_BallotOp : AGX_Op<"ballot"> {
  let summary = "Mask of the threads where a predicate is true";
  let description = [{
    Returns a mask with bit N set when thread N of the SIMD-group (or of the
    quad, in bits 0 to 3) is active and `predicate` is true there.

    ```mlir
    %mask = agx.ballot %pred : i32
    ```
  }];
  let arguments = (ins I1:$predicate,
                       DefaultValuedAttr<AGX_ScopeAttr,
                                         "::mlir::agx::Scope::simdgroup">:$scope);
  let results = (outs I32:$result);
  let assemblyFormat = "$predicate (`scope` $scope^)? attr-dict `:` type($result)";
}

def AGX_MemoryBarrierOp : AGX_Op<"memory_barrier"> {
  let summary = "Order the memory accesses of the thread";
  let description = [{
    Makes the memory accesses of the thread before the barrier visible to
    the other threads (of the threadgroup, or of the device) before those
    after it, like `__threadfence_block()` and `__threadfence()`.

    ```mlir
    agx.memory_barrier
    agx.memory_barrier scope threadgroup
    ```
  }];
  let arguments = (ins DefaultValuedAttr<AGX_MemoryScopeAttr,
                                         "::mlir::agx::MemoryScope::device">:$scope);
  let assemblyFormat = "(`scope` $scope^)? attr-dict";
}

def AGX_MatrixFragment : VectorOfLengthAndType<[2], [F16, BF16, F32]>;

def AGX_MMAOp : AGX_Op<"mma", [AllTypesMatch<["a", "b", "c", "result"]>]> {
  let summary = "8x8 matrix multiply-accumulate over the SIMD-group";
  let description = [{
    Computes D = A * B + C on 8x8 matrices whose elements are spread over
    the threads of the SIMD-group, like the MFMA operations of `amdgpu`.
    Thread t holds the elements (r, c) and (r, c + 1) of each matrix, with
    r = 4 * (t / 16) + (t / 2) % 4 and c = 4 * ((t / 8) % 2) + 2 * (t % 2).
    The products are accumulated in order of k with fused multiply-adds,
    starting from C.

    ```mlir
    %d = agx.mma %a * %b + %c : vector<2xf32>
    ```
  }];
  let arguments = (ins AGX_MatrixFragment:$a, AGX_MatrixFragment:$b,
                       AGX_MatrixFragment:$c);
  let results = (outs AGX_MatrixFragment:$result);
  let assemblyFormat = "$a `*` $b `+` $c attr-dict `:` type($result)";
}

def AGX_SetFPModeOp : AGX_Op<"set_fpmode"> {
  let summary = "Set the floating point mode";
  let description = [{
    Sets the rounding directions and denormal modes of the following
    constrained floating point operations (with dynamic rounding): bits 0-1
    are the float rounding direction (0 toward zero, 1 to nearest even, 2
    up, 3 down), bits 2-3 the half and double one, bit 4 flushes float
    denormals and bit 5 half and double denormals. The hardware rounds to
    nearest even only: the other directions are computed in software.

    ```mlir
    agx.set_fpmode %mode : i32
    ```
  }];
  let arguments = (ins I32:$mode);
  let assemblyFormat = "$mode attr-dict `:` type($mode)";
}

def AGX_GetFPModeOp : AGX_Op<"get_fpmode"> {
  let summary = "Read the floating point mode";
  let results = (outs I32:$mode);
  let assemblyFormat = "attr-dict `:` type($mode)";
}

#endif // AGX
