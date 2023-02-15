// RUN:   mlir-opt %s -pass-pipeline="builtin.module(async-func-to-async-runtime,async-to-async-runtime,func.func(async-runtime-ref-counting,async-runtime-ref-counting-opt),convert-async-to-llvm,func.func(convert-linalg-to-loops,convert-scf-to-cf),convert-linalg-to-llvm,convert-vector-to-llvm,func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext    \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_async_runtime%shlibext   \
// RUN: | FileCheck %s --dump-input=always

// FIXME: https://github.com/llvm/llvm-project/issues/57231
// UNSUPPORTED: hwasan

async.func @async_func_empty() -> !async.token {
  return
}

async.func @async_func_assert() -> !async.token {
  %false = arith.constant 0 : i1
  cf.assert %false, "error"
  return
}

async.func @async_func_nested_assert() -> !async.token {
  %token0 = async.call @async_func_assert() : () -> !async.token
  async.await %token0 : !async.token
  return
}

async.func @async_func_value_assert() -> !async.value<f32> {
  %false = arith.constant 0 : i1
  cf.assert %false, "error"
  %0 = arith.constant 123.45 : f32
  return %0 : f32
}

async.func @async_func_value_nested_assert() -> !async.value<f32> {
  %value0 = async.call @async_func_value_assert() : () -> !async.value<f32>
  %ret = async.await %value0 : !async.value<f32>
  return %ret : f32
}

async.func @async_func_return_value() -> !async.value<f32> {
  %0 = arith.constant 456.789 : f32
  return %0 : f32
}

async.func @async_func_non_blocking_await() -> !async.value<f32> {
  %value0 = async.call @async_func_return_value() : () -> !async.value<f32>
  %1 = async.await %value0 : !async.value<f32>
  return  %1 : f32
}

async.func @async_func_inside_memref() -> !async.value<memref<f32>> {
  %0 = memref.alloc() : memref<f32>
  %c0 = arith.constant 0.25 : f32
  memref.store %c0, %0[] : memref<f32>
  return %0 : memref<f32>
}

async.func @async_func_passed_memref(%arg0 : !async.value<memref<f32>>) -> !async.token {
  %unwrapped = async.await %arg0 : !async.value<memref<f32>>
  %0 = memref.load %unwrapped[] : memref<f32>
  %1 = arith.addf %0, %0 : f32
  memref.store %1, %unwrapped[] : memref<f32>
  return
}

async.func @async_execute_in_async_func(%arg0 : !async.value<memref<f32>>) -> !async.token {
  %token0 = async.execute {
    %unwrapped = async.await %arg0 : !async.value<memref<f32>>
    %0 = memref.load %unwrapped[] : memref<f32>
    %1 = arith.addf %0, %0 : f32
    memref.store %1, %unwrapped[] : memref<f32>
    async.yield
  }

  async.await %token0 : !async.token
  return
}


func.func @main() {
  %false = arith.constant 0 : i1

  // ------------------------------------------------------------------------ //
  // Check that simple async.func completes without errors.
  // ------------------------------------------------------------------------ //
  %token0 = async.call @async_func_empty() : () -> !async.token
  async.runtime.await %token0 : !async.token

  // CHECK: 0
  %err0 = async.runtime.is_error %token0 : !async.token
  vector.print %err0 : i1

  // ------------------------------------------------------------------------ //
  // Check that assertion in the async.func converted to async error.
  // ------------------------------------------------------------------------ //
  %token1 = async.call @async_func_assert() : () -> !async.token
  async.runtime.await %token1 : !async.token

  // CHECK: 1
  %err1 = async.runtime.is_error %token1 : !async.token
  vector.print %err1 : i1

  // ------------------------------------------------------------------------ //
  // Check error propagation from the nested async.func.
  // ------------------------------------------------------------------------ //
  %token2 = async.call @async_func_nested_assert() : () -> !async.token
  async.runtime.await %token2 : !async.token

  // CHECK: 1
  %err2 = async.runtime.is_error %token2 : !async.token
  vector.print %err2 : i1

  // ------------------------------------------------------------------------ //
  // Check error propagation from the nested async.func with async values.
  // ------------------------------------------------------------------------ //
  %value3 = async.call @async_func_value_nested_assert() : () -> !async.value<f32>
  async.runtime.await %value3 : !async.value<f32>

  // CHECK: 1
  %err3_0 = async.runtime.is_error %value3 : !async.value<f32>
  vector.print %err3_0 : i1

  // ------------------------------------------------------------------------ //
  // Non-blocking async.await inside the async.func
  // ------------------------------------------------------------------------ //
  %result0 = async.call @async_func_non_blocking_await() : () -> !async.value<f32>
  %4 = async.await %result0 : !async.value<f32>

  // CHECK: 456.789
  vector.print %4 : f32

  // ------------------------------------------------------------------------ //
  // Memref allocated inside async.func.
  // ------------------------------------------------------------------------ //
  %result1 = async.call @async_func_inside_memref() : () -> !async.value<memref<f32>>
  %5 = async.await %result1 : !async.value<memref<f32>>
  %6 = memref.cast %5 :  memref<f32> to memref<*xf32>

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT: [0.25]
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()

  // ------------------------------------------------------------------------ //
  // Memref passed as async.func parameter
  // ------------------------------------------------------------------------ //
  %token3 = async.call @async_func_passed_memref(%result1) : (!async.value<memref<f32>>) -> !async.token
  async.await %token3 : !async.token

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT: [0.5]
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()

  // ------------------------------------------------------------------------ //
  // async.execute inside async.func
  // ------------------------------------------------------------------------ //
  %token4 = async.call @async_execute_in_async_func(%result1) : (!async.value<memref<f32>>) -> !async.token
  async.await %token4 : !async.token

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT: [1]
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()

  memref.dealloc %5 : memref<f32>

  return
}

func.func private @printMemrefF32(memref<*xf32>)
  attributes { llvm.emit_c_interface }
