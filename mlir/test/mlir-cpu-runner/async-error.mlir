// RUN:   mlir-opt %s -pass-pipeline="async-to-async-runtime,func.func(async-runtime-ref-counting,async-runtime-ref-counting-opt),convert-async-to-llvm,func.func(convert-linalg-to-loops,convert-scf-to-cf),convert-linalg-to-llvm,convert-vector-to-llvm,func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts" \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext    \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_async_runtime%shlibext   \
// RUN: | FileCheck %s --dump-input=always

// FIXME: https://github.com/llvm/llvm-project/issues/57231
// UNSUPPORTED: hwasan

func.func @main() {
  %false = arith.constant 0 : i1

  // ------------------------------------------------------------------------ //
  // Check that simple async region completes without errors.
  // ------------------------------------------------------------------------ //
  %token0 = async.execute {
    async.yield
  }
  async.runtime.await %token0 : !async.token

  // CHECK: 0
  %err0 = async.runtime.is_error %token0 : !async.token
  vector.print %err0 : i1

  // ------------------------------------------------------------------------ //
  // Check that assertion in the async region converted to async error.
  // ------------------------------------------------------------------------ //
  %token1 = async.execute {
    cf.assert %false, "error"
    async.yield
  }
  async.runtime.await %token1 : !async.token

  // CHECK: 1
  %err1 = async.runtime.is_error %token1 : !async.token
  vector.print %err1 : i1

  // ------------------------------------------------------------------------ //
  // Check error propagation from the nested region.
  // ------------------------------------------------------------------------ //
  %token2 = async.execute {
    %token = async.execute {
      cf.assert %false, "error"
      async.yield
    }
    async.await %token : !async.token
    async.yield
  }
  async.runtime.await %token2 : !async.token

  // CHECK: 1
  %err2 = async.runtime.is_error %token2 : !async.token
  vector.print %err2 : i1

  // ------------------------------------------------------------------------ //
  // Check error propagation from the nested region with async values.
  // ------------------------------------------------------------------------ //
  %token3, %value3 = async.execute -> !async.value<f32> {
    %token, %value = async.execute -> !async.value<f32> {
      cf.assert %false, "error"
      %0 = arith.constant 123.45 : f32
      async.yield %0 : f32
    }
    %ret = async.await %value : !async.value<f32>
    async.yield %ret : f32
  }
  async.runtime.await %token3 : !async.token
  async.runtime.await %value3 : !async.value<f32>

  // CHECK: 1
  // CHECK: 1
  %err3_0 = async.runtime.is_error %token3 : !async.token
  %err3_1 = async.runtime.is_error %value3 : !async.value<f32>
  vector.print %err3_0 : i1
  vector.print %err3_1 : i1

  // ------------------------------------------------------------------------ //
  // Check error propagation from a token to the group.
  // ------------------------------------------------------------------------ //

  %c2 = arith.constant 2 : index
  %group0 = async.create_group %c2 : !async.group

  %token4 = async.execute {
    async.yield
  }

  %token5 = async.execute {
    cf.assert %false, "error"
    async.yield
  }

  %idx0 = async.add_to_group %token4, %group0 : !async.token
  %idx1 = async.add_to_group %token5, %group0 : !async.token

  async.runtime.await %group0 : !async.group

  // CHECK: 1
  %err4 = async.runtime.is_error %group0 : !async.group
  vector.print %err4 : i1

  return
}
