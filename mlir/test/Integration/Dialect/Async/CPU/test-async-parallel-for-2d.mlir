// RUN:   mlir-opt %s -async-parallel-for                                      \
// RUN:               -async-to-async-runtime                                  \
// RUN:               -async-runtime-ref-counting                              \
// RUN:               -async-runtime-ref-counting-opt                          \
// RUN:               -arith-expand                                            \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-scf-to-cf                                      \
// RUN:               -convert-memref-to-llvm                                  \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN:  -e entry -entry-point-result=void -O0                                 \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_async_runtime%shlibext\
// RUN: | FileCheck %s --dump-input=always

// RUN:   mlir-opt %s -async-parallel-for                                      \
// RUN:               -async-to-async-runtime                                  \
// RUN:               -async-runtime-policy-based-ref-counting                 \
// RUN:               -arith-expand                                            \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-scf-to-cf                                      \
// RUN:               -convert-memref-to-llvm                                  \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN:  -e entry -entry-point-result=void -O0                                 \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_async_runtime%shlibext\
// RUN: | FileCheck %s --dump-input=always

// RUN:   mlir-opt %s -async-parallel-for="async-dispatch=false                \
// RUN:                                    num-workers=20                      \
// RUN:                                    min-task-size=1"                    \
// RUN:               -async-to-async-runtime                                  \
// RUN:               -async-runtime-ref-counting                              \
// RUN:               -async-runtime-ref-counting-opt                          \
// RUN:               -arith-expand                                            \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-scf-to-cf                                      \
// RUN:               -convert-memref-to-llvm                                  \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN:  -e entry -entry-point-result=void -O0                                 \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_async_runtime%shlibext\
// RUN: | FileCheck %s --dump-input=always

func @entry() {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index

  %A = memref.alloc() : memref<8x8xf32>
  %U = memref.cast %A :  memref<8x8xf32> to memref<*xf32>

  // Initialize memref with zeros because we do load and store to in every test
  // to verify that we process each element of the iteration space once.
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    memref.store %c0, %A[%i, %j] : memref<8x8xf32>
  }

  // 1. (%i, %i) = (0, 8) to (8, 8) step (1, 1)
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    %0 = arith.muli %i, %c8 : index
    %1 = arith.addi %j, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%i, %j] : memref<8x8xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%i, %j] : memref<8x8xf32>
  }

  // CHECK:      [0, 1, 2, 3, 4, 5, 6, 7]
  // CHECK-NEXT: [8, 9, 10, 11, 12, 13, 14, 15]
  // CHECK-NEXT: [16, 17, 18, 19, 20, 21, 22, 23]
  // CHECK-NEXT: [24, 25, 26, 27, 28, 29, 30, 31]
  // CHECK-NEXT: [32, 33, 34, 35, 36, 37, 38, 39]
  // CHECK-NEXT: [40, 41, 42, 43, 44, 45, 46, 47]
  // CHECK-NEXT: [48, 49, 50, 51, 52, 53, 54, 55]
  // CHECK-NEXT: [56, 57, 58, 59, 60, 61, 62, 63]
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    memref.store %c0, %A[%i, %j] : memref<8x8xf32>
  }

  // 2. (%i, %i) = (0, 8) to (8, 8) step (2, 1)
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c2, %c1) {
    %0 = arith.muli %i, %c8 : index
    %1 = arith.addi %j, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%i, %j] : memref<8x8xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%i, %j] : memref<8x8xf32>
  }

  // CHECK:      [0, 1, 2, 3, 4, 5, 6, 7]
  // CHECK-NEXT: [0, 0, 0, 0, 0, 0, 0, 0]
  // CHECK-NEXT: [16, 17, 18, 19, 20, 21, 22, 23]
  // CHECK-NEXT: [0, 0, 0, 0, 0, 0, 0, 0]
  // CHECK-NEXT: [32, 33, 34, 35, 36, 37, 38, 39]
  // CHECK-NEXT: [0, 0, 0, 0, 0, 0, 0, 0]
  // CHECK-NEXT: [48, 49, 50, 51, 52, 53, 54, 55]
  // CHECK-NEXT: [0, 0, 0, 0, 0, 0, 0, 0]
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    memref.store %c0, %A[%i, %j] : memref<8x8xf32>
  }

  // 3. (%i, %i) = (0, 8) to (8, 8) step (1, 2)
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c2) {
    %0 = arith.muli %i, %c8 : index
    %1 = arith.addi %j, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%i, %j] : memref<8x8xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%i, %j] : memref<8x8xf32>
  }

  // CHECK:      [0, 0, 2, 0, 4, 0, 6, 0]
  // CHECK-NEXT: [8, 0, 10, 0, 12, 0, 14, 0]
  // CHECK-NEXT: [16, 0, 18, 0, 20, 0, 22, 0]
  // CHECK-NEXT: [24, 0, 26, 0, 28, 0, 30, 0]
  // CHECK-NEXT: [32, 0, 34, 0, 36, 0, 38, 0]
  // CHECK-NEXT: [40, 0, 42, 0, 44, 0, 46, 0]
  // CHECK-NEXT: [48, 0, 50, 0, 52, 0, 54, 0]
  // CHECK-NEXT: [56, 0, 58, 0, 60, 0, 62, 0]
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  memref.dealloc %A : memref<8x8xf32>

  return
}

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
