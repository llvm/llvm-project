//===-- Passes.td - AMDGPU pass declarations ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the AMDGPU Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_TRANSFORMS_PASSES_TD_
#define MLIR_DIALECT_AMDGPU_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AmdgpuEmulateAtomicsPass : Pass<"amdgpu-emulate-atomics"> {
  let summary = "Emulate atomic operations the target does not support";
  let description = [{
    This pass rewrites any AMDGPU-specific atomic operation that the target does
    not support into a compare-and-swap loop.
  }];
  let dependentDialects = [
    "cf::ControlFlowDialect",
    "arith::ArithDialect",
    "vector::VectorDialect"
  ];
  let options = [
    Option<"arch", "arch", "std::string",
           /*default=*/"\"invalid\"",
           "Target architecture, as in Clang, with optional target-ID modifiers. New-style triples such as amdgpu9.42-amd-amdhsa are preferred, and can be extended to a full target ID like amdgpu9.42-amd-amdhsa--gfx942:xnack+:sramecc-. Bare chip names like gfx1250 or gfx942:xnack- are also supported. Defaults to an invalid target so that one must be given explicitly">,
    Option<"chipset", "chipset", "std::string", /*default=*/"\"\"",
           "Deprecated alias for 'arch'.">,
  ];
}

def AmdgpuResolveStridedMetadataPass : Pass<"amdgpu-resolve-strided-metadata"> {
  let summary = "Resolve memref.extract_strided_metadata on AMDGPU ops";
  let description = [{
    This pass rrewrites `memref.extract_strided_metadata` operations
    targeting the AMDGPU dialect casts.

    The patterns in this pass should normally be run alongside those in
    -expand-strided-metadata, and creating a pass that combines those two
    sets of patterns is the recommended way to use this functionality.
    However, this pass (which will likely need a second -expand-strided-metadata
    after it) is provided so that simple usecases do not need to create custom passes.
    These patterns have not been added to -expnad-strided-metadata to
    prevent the memref dialect from depending on platform-specific code.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect"
  ];
}

def AmdgpuMaskedloadToLoadPass : Pass<"amdgpu-maskedload-to-load"> {
  let summary = "Lower the operations from the vector maskedload to vector load";
  let description = [{
    This pass creates a transfer read op lowering optimization. The lowering
    will produce a conditional check at runtime. If within bounds, a vector
    trasfer read op will be lowered to a combination of vector.load, arith.select
    and vector.broadcast. If not, it will fallback to the default lowering
    of the transfer_read op.

    This pattern will make it possible for masked transfer_read to be lowered
    towards buffer load with bounds check, allowing a more optimized global
    load accessing pattern compared with existing implementation of
    llvm.intr.masked.load on vectors.
  }];
  let dependentDialects = [
    "scf::SCFDialect",
    "memref::MemRefDialect"
  ];
}

#endif // MLIR_DIALECT_AMDGPU_TRANSFORMS_PASSES_TD_
