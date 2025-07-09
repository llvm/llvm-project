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
  let summary = "Emulate atomic operations on chipsets that do not support them";
  let description = [{
    This pass rewrites any AMDGPU-specific atomic operation that is not supported
    on the given `chipset` into a compare-and-swap loop.
  }];
  let dependentDialects = [
    "cf::ControlFlowDialect",
    "arith::ArithDialect",
    "vector::VectorDialect"
  ];
  let options = [Option<"chipset", "chipset", "std::string",
                        /*default=*/"\"gfx000\"",
                        "Chipset that these operations will run on">];
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
