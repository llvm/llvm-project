// RUN:   mlir-opt %s -pass-pipeline="builtin.module(async-to-async-runtime,func.func(async-runtime-ref-counting,async-runtime-ref-counting-opt),convert-async-to-llvm,func.func(convert-arith-to-llvm),convert-vector-to-llvm,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext    \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_async_runtime%shlibext   \
// RUN: | FileCheck %s --dump-input=always

// FIXME: https://github.com/llvm/llvm-project/issues/57231
// UNSUPPORTED: hwasan

func.func @main() {

  // ------------------------------------------------------------------------ //
  // Blocking async.await outside of the async.execute.
  // ------------------------------------------------------------------------ //
  %token, %result = async.execute -> !async.value<f32> {
    %0 = arith.constant 123.456 : f32
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
      %2 = arith.constant 456.789 : f32
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
    %5 = memref.alloc() : memref<f32>
    %c0 = arith.constant 0.25 : f32
    memref.store %c0, %5[]: memref<f32>
    async.yield %5 : memref<f32>
  }
  %6 = async.await %result2 : !async.value<memref<f32>>
  %7 = memref.cast %6 :  memref<f32> to memref<*xf32>

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT: [0.25]
  call @printMemrefF32(%7): (memref<*xf32>) -> ()

  // ------------------------------------------------------------------------ //
  // Memref passed as async.execute operand.
  // ------------------------------------------------------------------------ //
  %token3 = async.execute(%result2 as %unwrapped : !async.value<memref<f32>>) {
    %8 = memref.load %unwrapped[]: memref<f32>
    %9 = arith.addf %8, %8 : f32
    memref.store %9, %unwrapped[]: memref<f32>
    async.yield
  }
  async.await %token3 : !async.token

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT: [0.5]
  call @printMemrefF32(%7): (memref<*xf32>) -> ()

  memref.dealloc %6 : memref<f32>

  return
}

func.func private @printMemrefF32(memref<*xf32>)
  attributes { llvm.emit_c_interface }
