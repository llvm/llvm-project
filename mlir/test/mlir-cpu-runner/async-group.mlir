// RUN:   mlir-opt %s -convert-async-to-llvm                                   \
// RUN:               -convert-std-to-llvm                                     \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext    \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_async_runtime%shlibext   \
// RUN: | FileCheck %s

func @main() {
  %group = async.create_group

  %token0 = async.execute { async.yield }
  %token1 = async.execute { async.yield }
  %token2 = async.execute { async.yield }
  %token3 = async.execute { async.yield }
  %token4 = async.execute { async.yield }

  %0 = async.add_to_group %token0, %group : !async.token
  %1 = async.add_to_group %token1, %group : !async.token
  %2 = async.add_to_group %token2, %group : !async.token
  %3 = async.add_to_group %token3, %group : !async.token
  %4 = async.add_to_group %token4, %group : !async.token

  %token5 = async.execute {
    async.await_all %group
    async.yield
  }

  %group0 = async.create_group
  %5 = async.add_to_group %token5, %group0 : !async.token
  async.await_all %group0

  // CHECK: Current thread id: [[THREAD:.*]]
  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()

  return
}

func @mlirAsyncRuntimePrintCurrentThreadId() -> ()
