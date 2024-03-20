// REQUIRES: host-supports-nvptx
// RUN: mlir-opt %s \
// RUN:  | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
// RUN:   | FileCheck %s

// RUN: mlir-opt %s \
// RUN:  | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
// RUN:    --mlir-print-ir-after=convert-gpu-to-nvvm 2>&1 \
// RUN:  | FileCheck %s --check-prefixes=CHECK-NVVM

// This test checks whether the GPU region is compiled correctly to PTX by 
// pipeline. It doesn't test IR for GPU side, but it can test Host IR and 
// generated PTX.

// CHECK-LABEL: llvm.func @test_math(%arg0: f32) {
func.func @test_math(%arg0 : f32) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    // CHECK: gpu.launch_func  @test_math_kernel::@test_math_kernel
    // CHECK: gpu.binary @test_math_kernel  [#gpu.object<#nvvm.target
    gpu.launch 
        blocks(%0, %1, %2) in (%3 = %c1, %4 = %c1, %5 = %c1) 
        threads(%6, %7, %8) in (%9 = %c2, %10 = %c1, %11 = %c1) { 
        // CHECK-NVVM: __nv_expf 
        %s1 = math.exp %arg0 : f32
        gpu.printf "%f" %s1 : f32
        gpu.terminator
    }
    return
}