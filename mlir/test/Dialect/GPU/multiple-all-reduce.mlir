// RUN: mlir-opt --gpu-kernel-outlining --convert-gpu-to-nvvm %s | FileCheck %s

func @main() {
  %data = alloc() : memref<2x6xf32>
  %sum = alloc() : memref<2xf32>
  %mul = alloc() : memref<2xf32>
  %c1 = constant 1 : index

  // ADD + MUL
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %val = load %data[%bx, %tx] : memref<2x6xf32>
    %reduced0 = "gpu.all_reduce"(%val) ({}) { op = "add" } : (f32) -> (f32)
    store %reduced0, %sum[%bx] : memref<2xf32>
    %reduced1 = "gpu.all_reduce"(%val) ({}) { op = "mul" } : (f32) -> (f32)
    store %reduced1, %mul[%bx] : memref<2xf32>
    gpu.terminator
  }

// CHECK:      gpu.module @main_kernel {
// CHECK-NEXT:   llvm.mlir.global internal @{{.*}}() {addr_space = 3 : i32} : !llvm.array<32 x f32>
// CHECK-NEXT:   llvm.mlir.global internal @{{.*}}() {addr_space = 3 : i32} : !llvm.array<32 x f32>

  return
}
