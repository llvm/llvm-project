// RUN: mlir-opt %s -convert-linalg-to-std | FileCheck %s

func private @print_memref_f32(memref<*xf32>)

// CHECK:  func private @linalg_fill_f32_viewsxsxf32(f32, memref<?x?xf32, {{.*}}>) attributes {llvm.emit_c_interface}
// CHECK:  func private @linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32(memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>) attributes {llvm.emit_c_interface}

func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>) -> (memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %x = memref.dim %A, %c0 : memref<?x?xf32>
  %y = memref.dim %B, %c1 : memref<?x?xf32>
  %C = memref.alloc(%x, %y) : memref<?x?xf32>

  // CHECK: call @linalg_fill_f32_viewsxsxf32({{.*}}) : (f32, memref<?x?xf32, {{.*}}>)
  linalg.fill(%f0, %C) : f32, memref<?x?xf32>

  // CHECK:  call @linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32({{.*}}) : (memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>) -> ()
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
                outs(%C: memref<?x?xf32>)
  return %C : memref<?x?xf32>
}
  
