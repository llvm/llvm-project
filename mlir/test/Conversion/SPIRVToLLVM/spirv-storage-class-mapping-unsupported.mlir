// RUN: mlir-opt -convert-spirv-to-llvm='client-api=Metal' -verify-diagnostics %s
// RUN: mlir-opt -convert-spirv-to-llvm='client-api=Vulkan' -verify-diagnostics %s
// RUN: mlir-opt -convert-spirv-to-llvm='client-api=WebGPU' -verify-diagnostics %s

module {}  // expected-warning-re {{address space mapping for client '{{.*}}' not implemented}}
