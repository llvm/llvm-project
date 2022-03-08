// RUN:   mlir-opt %s                                                          \
// RUN:               -convert-linalg-to-parallel-loops                        \
// RUN:               -async-parallel-for                                      \
// RUN:               -async-to-async-runtime                                  \
// RUN:               -async-runtime-ref-counting                              \
// RUN:               -async-runtime-ref-counting-opt                          \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-scf-to-cf                                      \
// RUN:               -arith-expand                                            \
// RUN:               -memref-expand                                              \
// RUN:               -convert-vector-to-llvm                                  \
// RUN:               -convert-memref-to-llvm                                  \
// RUN:               -convert-func-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN: -e entry -entry-point-result=void -O3                                  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext\
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_async_runtime%shlibext \
// RUN: | FileCheck %s --dump-input=always

// RUN:   mlir-opt %s                                                          \
// RUN:               -convert-linalg-to-loops                                 \
// RUN:               -convert-scf-to-cf                                      \
// RUN:               -convert-vector-to-llvm                                  \
// RUN:               -convert-memref-to-llvm                                  \
// RUN:               -convert-func-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN: -e entry -entry-point-result=void -O3                                  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext\
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_async_runtime%shlibext \
// RUN: | FileCheck %s --dump-input=always

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func @linalg_generic(%lhs: memref<?x?xf32>,
                     %rhs: memref<?x?xf32>,
                     %sum: memref<?x?xf32>) {
  linalg.generic {
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"]
  }
    ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>)
    outs(%sum : memref<?x?xf32>)
  {
    ^bb0(%lhs_in: f32, %rhs_in: f32, %sum_out: f32):
      %0 = arith.addf %lhs_in, %rhs_in : f32
      linalg.yield %0 : f32
  }

  return
}

func @entry() {
  %f1 = arith.constant 1.0 : f32
  %f4 = arith.constant 4.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cN = arith.constant 50 : index

  //
  // Sanity check for the function under test.
  //

  %LHS10 = memref.alloc() {alignment = 64} : memref<1x10xf32>
  %RHS10 = memref.alloc() {alignment = 64} : memref<1x10xf32>
  %DST10 = memref.alloc() {alignment = 64} : memref<1x10xf32>

  linalg.fill(%f1, %LHS10) : f32, memref<1x10xf32>
  linalg.fill(%f1, %RHS10) : f32, memref<1x10xf32>

  %LHS = memref.cast %LHS10 : memref<1x10xf32> to memref<?x?xf32>
  %RHS = memref.cast %RHS10 : memref<1x10xf32> to memref<?x?xf32>
  %DST = memref.cast %DST10 : memref<1x10xf32> to memref<?x?xf32>

  call @linalg_generic(%LHS, %RHS, %DST)
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

  // CHECK: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  %U = memref.cast %DST10 :  memref<1x10xf32> to memref<*xf32>
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  memref.dealloc %LHS10: memref<1x10xf32>
  memref.dealloc %RHS10: memref<1x10xf32>
  memref.dealloc %DST10: memref<1x10xf32>

  //
  // Allocate data for microbenchmarks.
  //

  %LHS1024 = memref.alloc() {alignment = 64} : memref<1024x1024xf32>
  %RHS1024 = memref.alloc() {alignment = 64} : memref<1024x1024xf32>
  %DST1024 = memref.alloc() {alignment = 64} : memref<1024x1024xf32>

  %LHS0 = memref.cast %LHS1024 : memref<1024x1024xf32> to memref<?x?xf32>
  %RHS0 = memref.cast %RHS1024 : memref<1024x1024xf32> to memref<?x?xf32>
  %DST0 = memref.cast %DST1024 : memref<1024x1024xf32> to memref<?x?xf32>

  //
  // Warm up.
  //

  call @linalg_generic(%LHS0, %RHS0, %DST0)
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

  //
  // Measure execution time.
  //

  %t0 = call @rtclock() : () -> f64
  scf.for %i = %c0 to %cN step %c1 {
    call @linalg_generic(%LHS0, %RHS0, %DST0)
      : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  }
  %t1 = call @rtclock() : () -> f64
  %t1024 = arith.subf %t1, %t0 : f64

  // Print timings.
  vector.print %t1024 : f64

  // Free.
  memref.dealloc %LHS1024: memref<1024x1024xf32>
  memref.dealloc %RHS1024: memref<1024x1024xf32>
  memref.dealloc %DST1024: memref<1024x1024xf32>

  return
}

func private @rtclock() -> f64

func private @print_memref_f32(memref<*xf32>)
  attributes { llvm.emit_c_interface }
