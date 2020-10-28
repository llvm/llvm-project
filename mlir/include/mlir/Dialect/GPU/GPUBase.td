//===-- GPUBase.td - GPU dialect definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the GPU dialect
//
//===----------------------------------------------------------------------===//

#ifndef GPU_BASE
#define GPU_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU Dialect.
//===----------------------------------------------------------------------===//

def GPU_Dialect : Dialect {
  let name = "gpu";
  let cppNamespace = "::mlir::gpu";
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate the modules that contain
    /// kernel modules.
    static StringRef getContainerModuleAttrName() {
      return "gpu.container_module";
    }
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "gpu.kernel"; }

    /// Returns whether the given function is a kernel function, i.e., has the
    /// 'gpu.kernel' attribute.
    static bool isKernel(Operation *op);

    /// Returns the number of workgroup (thread, block) dimensions supported in
    /// the GPU dialect.
    // TODO: consider generalizing this.
    static unsigned getNumWorkgroupDimensions() { return 3; }

    /// Returns the numeric value used to identify the workgroup memory address
    /// space.
    static unsigned getWorkgroupAddressSpace() { return 3; }

    /// Returns the numeric value used to identify the private memory address
    /// space.
    static unsigned getPrivateAddressSpace() { return 5; }
  }];
}

#endif // GPU_BASE
