// RUN: mlir-opt -split-input-file --openacc-legalize-data-values %s | FileCheck %s --check-prefixes=CHECK,DEVICE
// RUN: mlir-opt -split-input-file --openacc-legalize-data-values=host-to-device=false %s | FileCheck %s --check-prefixes=CHECK,HOST

func.func @test(%a: memref<10xf32>) {
  %devptr = acc.use_device varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.host_data dataOperands(%devptr : memref<10xf32>) {
    func.call @foo(%a) : (memref<10xf32>) -> ()
    acc.terminator
  }
  return
}
func.func private @foo(memref<10xf32>)

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>)
// CHECK: %[[USE_DEVICE:.*]] = acc.use_device varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.host_data dataOperands(%[[USE_DEVICE]] : memref<10xf32>) {
// DEVICE:   func.call @foo(%[[USE_DEVICE]]) : (memref<10xf32>) -> ()
// HOST:     func.call @foo(%[[A]]): (memref<10xf32>) -> ()
// CHECK:   acc.terminator
// CHECK: }