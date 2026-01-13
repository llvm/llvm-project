// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: @test = internal global [1 x float] [float -0.000000e+00]
llvm.mlir.global internal @test(dense<-0.000000e+00> : tensor<1xf32>) {addr_space = 0 : i32} : !llvm.array<1 x f32>
