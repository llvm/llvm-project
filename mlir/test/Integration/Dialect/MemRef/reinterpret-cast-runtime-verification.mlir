// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -lower-affine \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -test-cf-assert \
// RUN:     -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @reinterpret_cast(%memref: memref<1xf32>, %offset: index) {
    memref.reinterpret_cast %memref to
                    offset: [%offset],
                    sizes: [1],
                    strides: [1]
                  : memref<1xf32> to  memref<1xf32, strided<[1], offset: ?>>
    return
}

func.func @reinterpret_cast_fully_dynamic(%memref: memref<?xf32>, %offset: index, %size: index, %stride: index)  {
    memref.reinterpret_cast %memref to
                    offset: [%offset],
                    sizes: [%size],
                    strides: [%stride]
                  : memref<?xf32> to  memref<?xf32, strided<[?], offset: ?>>
    return
}

func.func @main() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %n1 = arith.constant -1 : index
  %4 = arith.constant 4 : index
  %5 = arith.constant 5 : index

  %alloca_1 = memref.alloca() : memref<1xf32>
  %alloca_4 = memref.alloca() : memref<4xf32>
  %alloca_4_dyn = memref.cast %alloca_4 : memref<4xf32> to memref<?xf32>

  // Offset is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.reinterpret_cast"(%{{.*}})
  // CHECK-NEXT: ^ result of reinterpret_cast is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @reinterpret_cast(%alloca_1, %1) : (memref<1xf32>, index) -> ()

  // Offset is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.reinterpret_cast"(%{{.*}})
  // CHECK-NEXT: ^ result of reinterpret_cast is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @reinterpret_cast(%alloca_1, %n1) : (memref<1xf32>, index) -> ()

  // Size is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.reinterpret_cast"(%{{.*}})
  // CHECK-NEXT: ^ result of reinterpret_cast is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @reinterpret_cast_fully_dynamic(%alloca_4_dyn, %0, %5, %1) : (memref<?xf32>, index, index, index) -> ()

  // Stride is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.reinterpret_cast"(%{{.*}})
  // CHECK-NEXT: ^ result of reinterpret_cast is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @reinterpret_cast_fully_dynamic(%alloca_4_dyn, %0, %4, %4) : (memref<?xf32>, index, index, index) -> ()

  //  CHECK-NOT: ERROR: Runtime op verification failed
  func.call @reinterpret_cast(%alloca_1, %0) : (memref<1xf32>, index) -> ()

  //  CHECK-NOT: ERROR: Runtime op verification failed
  func.call @reinterpret_cast_fully_dynamic(%alloca_4_dyn, %0, %4, %1) : (memref<?xf32>, index, index, index) -> ()

  return
}
