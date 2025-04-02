// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx942 | FileCheck %s

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3

// CHECK-LABEL: func @global_load_to_rocdl_f32
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xf32, 1>)
func.func @global_load_to_rocdl_f32(%global : memref<128x72xf32, #gpu_global_addrspace>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu_lds_addrspace>
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<128x72xf32, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x64xf32, 3>
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : memref<64x64xf32, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1] 
  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]
  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]]
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: rocdl.global.load.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], %[[C4]], %[[C0]], %[[C0_2]]
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0] {transferType = f32} : memref<128x72xf32, #gpu_global_addrspace>, memref<64x64xf32, #gpu_lds_addrspace>
  func.return
}
