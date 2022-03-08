// RUN:   mlir-opt %s -pass-pipeline="async-to-async-runtime,func.func(async-runtime-ref-counting,async-runtime-ref-counting-opt),convert-async-to-llvm,func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts" \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext    \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_async_runtime%shlibext   \
// RUN: | FileCheck %s

// This is crashing in CI "most of the time" on a AMD Rome CPU VM on GCP with:
//    Tracer caught signal 11: addr=0x7a800028 pc=0x2e81ba sp=0x7efd2a7ffd50
//    LeakSanitizer has encountered a fatal error.
// This is hard to reproduce locally unfortunately. Disable it with ASAN/LSAN
// to keep the bot green for now.
// UNSUPPORTED: asan

func @main() {
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  %group = async.create_group %c5 : !async.group

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

  %group0 = async.create_group %c1 : !async.group
  %5 = async.add_to_group %token5, %group0 : !async.token
  async.await_all %group0

  // CHECK: Current thread id: [[THREAD:.*]]
  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()

  return
}

func private @mlirAsyncRuntimePrintCurrentThreadId() -> ()
