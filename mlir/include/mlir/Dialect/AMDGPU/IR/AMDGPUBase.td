//===-- AMDGPUBase.td - AMDGPU dialect base *- tablegen -*-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_IR_AMDGPUBASE_TD
#define MLIR_DIALECT_AMDGPU_IR_AMDGPUBASE_TD

include "mlir/IR/DialectBase.td"

def AMDGPU_Dialect : Dialect {
  let name = "amdgpu";
  let cppNamespace = "::mlir::amdgpu";
  let description = [{
    The `AMDGPU` dialect provides wrappers around AMD-specific functionality
    and LLVM intrinsics. These wrappers should be used in conjunction with
    more generic dialects, such as `gpu` and `vector`, when generating LLVM IR
    that will eventually be executed on AMD hardware.

    # What goes here?
    In many cases, AMD GPU functionality can be accessed either though generic
    operations (such as those in the `gpu`, `vector`, or `math`) or through
    the `rocdl` dialect's intrinsic wrappers. However, there are instances where
    AMD-specific functionally benefits from a wrapper around the underlying
    LLVM intrinsics.

    In general terms, operations or types should be added to this dialect when they
    wrap some AMD-specific functionality in a way that makes it work better with the
    MLIR ecosystem and its types or when those buitins would be needlessly
    complex to work with (such as if they features magic constants at the LLVM level).

    An additional set of operations that belong in this dialect are those that
    have chipset-specific differences that can be abstracted over in a useful way.

    To give some concrete examples:

    - `amdgpu.mfma` and `amdgpu.wmma` exist in order to make a large set of
      intrinsics more compatible with the MLIR type system (such as by allowing
      8-bit float vectors to be passed as `vector<N x f8E4M3FN>` or
      `vector<N x f8E4M2>` instead of as packed 32-bit integers whose element type
      is controlled by separate operator-level constants. These operations also
      allow the same `amdgpu.mfma` operation to be used regardless of the target
      chip.
    - `amdgpu.swizzle_bitmode` provides a wrapper around the `ds.swizzle` intrinsic,
      allowing a wider range of types (such as `vector<2xf16>`) to be used natively
      and eliminating the need to pack the and, or, and xor components using opaque
      shifts.
    - Operations like `amdgpu.gather_to_lds` provide `memref`-ized wrappers around
      intrinsics that take a pointer, and are nontrivial enough to justify inclusion
      in this dialect.


    Note that simple intrinsics like `rocdl.sin` or `rocdl.s.barrier` should not
    receive wrapper operations, as nothing is gained from the duplicate operation.
    As a rule of thumb, if an operation's rewrite in AMDGPUToROCDL would be only
    a `replaceOpWithNewOp` call, no AMDGPU dialect operation is needed.

    # Design guidelines

    Operations should leverage MLIR's "standard" types where possible. MLIR has
    a more extensible type system than LLVM (especially in the area of small floats)
    and those types should be used to create more ergonomic wrappers. In particular,
    intrinsics that take pointers should have wrappers in this dialect that take
    `memref` arguments and indices.

    Operations should use properties or attributes in cases where the underlying
    intrinsic uses `immarg`s (except in cases where that attribute can be represented
    in the type system).

    If it is possible to generalize the types of an operation, it should be done.
    For example, the underlying operations for permutations and swizzles always
    take 32-bit operands. Their AMDGPU wrappers can take any type, and will apply
    padding and expansion to multiple instructions as needed. This makes these
    operations easier to target because it hides the bitcasts and extracts
    until the final lowering.

    When the underlying operation uses magic constants, those should be presented
    in a more programmer-friendly fashion, such as through enums or though
    using separate arguments that are later combined. (For example, see the
    design of the `amdgpu.dpp` and `amdgpu.fat_raw_buffer_cast` operations.)

    If sufficiently similar functionality on multiple hardware generations can be
    encapsulated into a single operation, it should be done. The lowering to
    intrinsics should either throw an error when an unsupported capability is
    used or ignore it. Which of these is two failure modes is more appropriate
    depends on the nature of the feature, but errors are a safe default choice.

    # Documentation guidelines

    AMDGPU dialect operations should document how any abstractions they introduce
    translate to LLVM intrinsics or hardware operations.

    While documenting the semantics of the underlying operations is not required,
    is preferred to provide an overview of the operation's functionality,
    especially in cases where the documentation is widely distributed. Someone
    looking at an AMDGPU dialect operation should be able to generally understand
    what it does and have found the keywords they'll need for more detail.

    Operation documentation should include usage examples.

    Note that this dialect uses LLVM's gfx numbers to refer to individual
    architectures/chipsets and not product names or codenames.
  }];


  let dependentDialects = [
    "ROCDL::ROCDLDialect",
    "arith::ArithDialect",
    "gpu::GPUDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // MLIR_DIALECT_AMDGPU_IR_AMDGPUBASE_TD
