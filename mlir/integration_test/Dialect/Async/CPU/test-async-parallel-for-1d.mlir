// RUN:   mlir-opt %s -async-parallel-for                                      \
// RUN:               -async-ref-counting                                      \
// RUN:               -async-to-async-runtime                                  \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-scf-to-std                                      \
// RUN:               -convert-std-to-llvm                                     \
// RUN: | mlir-cpu-runner                                                      \
// RUN:  -e entry -entry-point-result=void -O0                                 \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_async_runtime%shlibext\
// RUN: | FileCheck %s --dump-input=always

func @entry() {
  %c0 = constant 0.0 : f32
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index

  %lb = constant 0 : index
  %ub = constant 9 : index

  %A = alloc() : memref<9xf32>
  %U = memref_cast %A :  memref<9xf32> to memref<*xf32>

  // 1. %i = (0) to (9) step (1)
  scf.parallel (%i) = (%lb) to (%ub) step (%c1) {
    %0 = index_cast %i : index to i32
    %1 = sitofp %0 : i32 to f32
    store %1, %A[%i] : memref<9xf32>
  }
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  scf.parallel (%i) = (%lb) to (%ub) step (%c1) {
    store %c0, %A[%i] : memref<9xf32>
  }

  // 2. %i = (0) to (9) step (2)
  scf.parallel (%i) = (%lb) to (%ub) step (%c2) {
    %0 = index_cast %i : index to i32
    %1 = sitofp %0 : i32 to f32
    store %1, %A[%i] : memref<9xf32>
  }
  // CHECK:  [0, 0, 2, 0, 4, 0, 6, 0, 8]
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  scf.parallel (%i) = (%lb) to (%ub) step (%c1) {
    store %c0, %A[%i] : memref<9xf32>
  }

  // 3. %i = (-20) to (-11) step (3)
  %lb0 = constant -20 : index
  %ub0 = constant -11 : index
  scf.parallel (%i) = (%lb0) to (%ub0) step (%c3) {
    %0 = index_cast %i : index to i32
    %1 = sitofp %0 : i32 to f32
    %2 = constant 20 : index
    %3 = addi %i, %2 : index
    store %1, %A[%3] : memref<9xf32>
  }
  // CHECK: [-20, 0, 0, -17, 0, 0, -14, 0, 0]
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  dealloc %A : memref<9xf32>
  return
}

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
