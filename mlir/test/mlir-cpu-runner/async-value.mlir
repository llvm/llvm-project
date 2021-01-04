// RUN:   mlir-opt %s -async-ref-counting                                      \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-vector-to-llvm                                  \
// RUN:               -convert-std-to-llvm                                     \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext    \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_async_runtime%shlibext   \
// RUN: | FileCheck %s --dump-input=always

func @main() {

  // ------------------------------------------------------------------------ //
  // Blocking async.await outside of the async.execute.
  // ------------------------------------------------------------------------ //
  %token, %result = async.execute -> !async.value<f32> {
    %0 = constant 123.456 : f32
    async.yield %0 : f32
  }
  %1 = async.await %result : !async.value<f32>

  // CHECK: 123.456
  vector.print %1 : f32

  // ------------------------------------------------------------------------ //
  // Non-blocking async.await inside the async.execute
  // ------------------------------------------------------------------------ //
  %token0, %result0 = async.execute -> !async.value<f32> {
    %token1, %result2 = async.execute -> !async.value<f32> {
      %2 = constant 456.789 : f32
      async.yield %2 : f32
    }
    %3 = async.await %result2 : !async.value<f32>
    async.yield %3 : f32
  }
  %4 = async.await %result0 : !async.value<f32>

  // CHECK: 456.789
  vector.print %4 : f32

  // ------------------------------------------------------------------------ //
  // Memref allocated inside async.execute region.
  // ------------------------------------------------------------------------ //
  %token2, %result2 = async.execute[%token0] -> !async.value<memref<f32>> {
    %5 = alloc() : memref<f32>
    %c0 = constant 0.25 : f32
    store %c0, %5[]: memref<f32>
    async.yield %5 : memref<f32>
  }
  %6 = async.await %result2 : !async.value<memref<f32>>
  %7 = memref_cast %6 :  memref<f32> to memref<*xf32>

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT: [0.25]
  call @print_memref_f32(%7): (memref<*xf32>) -> ()

  // ------------------------------------------------------------------------ //
  // Memref passed as async.execute operand.
  // ------------------------------------------------------------------------ //
  %token3 = async.execute(%result2 as %unwrapped : !async.value<memref<f32>>) {
    %8 = load %unwrapped[]: memref<f32>
    %9 = addf %8, %8 : f32
    store %9, %unwrapped[]: memref<f32>
    async.yield
  }
  async.await %token3 : !async.token

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT: [0.5]
  call @print_memref_f32(%7): (memref<*xf32>) -> ()

  dealloc %6 : memref<f32>

  return
}

func private @print_memref_f32(memref<*xf32>)
  attributes { llvm.emit_c_interface }
