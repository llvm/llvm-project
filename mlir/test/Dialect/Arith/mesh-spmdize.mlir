// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(mesh-spmdization))" \
// RUN:   %s | FileCheck %s

mesh.mesh @mesh4x4(shape = 4x4)

// CHECK-LABEL: func @test_spmdize_constant
// CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> :
// tensor<256x1024xf32> CHECK-NEXT: [[vc434_i32:%.*]] = arith.constant 434 :
// i32 CHECK-NEXT: return [[vcst]] : tensor<256x1024xf32>
func.func @test_spmdize_constant() ->(tensor<1024x1024xf32>)attributes{llvm.emit_c_interface} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %sharding_1 = mesh.sharding @mesh4x4 split_axes = [[0]] : !mesh.sharding
  %sharding_annotated_1 = mesh.shard %cst to %sharding_1 : tensor<1024x1024xf32>
  %ci = arith.constant 434 : i32
  return %sharding_annotated_1 : tensor<1024x1024xf32>
}
