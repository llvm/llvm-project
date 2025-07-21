// RUN:   mlir-opt %s -pass-pipeline="builtin.module(async-to-async-runtime,func.func(async-runtime-ref-counting,async-runtime-ref-counting-opt),convert-async-to-llvm,func.func(convert-linalg-to-loops,convert-scf-to-cf),finalize-memref-to-llvm,func.func(convert-arith-to-llvm),convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_c_runner_utils  \
// RUN:     -shared-libs=%mlir_runner_utils    \
// RUN:     -shared-libs=%mlir_async_runtime   \
// RUN: | FileCheck %s

// FIXME: https://github.com/llvm/llvm-project/issues/57231
// UNSUPPORTED: asan
// UNSUPPORTED: hwasan
// FIXME: Windows does not have aligned_alloc
// UNSUPPORTED: system-windows

func.func @main() {
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %i3 = arith.constant 3 : index

  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %c4 = arith.constant 4.0 : f32

  %A = memref.alloc() : memref<4xf32>
  linalg.fill ins(%c0 : f32) outs(%A : memref<4xf32>)

  // CHECK: [0, 0, 0, 0]
  %U = memref.cast %A :  memref<4xf32> to memref<*xf32>
  call @printMemrefF32(%U): (memref<*xf32>) -> ()

  // CHECK: Current thread id: [[MAIN:.*]]
  // CHECK: [1, 0, 0, 0]
  memref.store %c1, %A[%i0]: memref<4xf32>
  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
  call @printMemrefF32(%U): (memref<*xf32>) -> ()

  %outer = async.execute {
    // CHECK: Current thread id: [[THREAD0:.*]]
    // CHECK: [1, 2, 0, 0]
    memref.store %c2, %A[%i1]: memref<4xf32>
    func.call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    func.call @printMemrefF32(%U): (memref<*xf32>) -> ()

    // No op async region to create a token for testing async dependency.
    %noop = async.execute {
      // CHECK: Current thread id: [[THREAD1:.*]]
      func.call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
      async.yield
    }

    %inner = async.execute [%noop] {
      // CHECK: Current thread id: [[THREAD2:.*]]
      // CHECK: [1, 2, 3, 0]
      memref.store %c3, %A[%i2]: memref<4xf32>
      func.call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
      func.call @printMemrefF32(%U): (memref<*xf32>) -> ()

      async.yield
    }
    async.await %inner : !async.token

    // CHECK: Current thread id: [[THREAD3:.*]]
    // CHECK: [1, 2, 3, 4]
    memref.store %c4, %A[%i3]: memref<4xf32>
    func.call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    func.call @printMemrefF32(%U): (memref<*xf32>) -> ()

    async.yield
  }
  async.await %outer : !async.token

  // CHECK: Current thread id: [[MAIN]]
  // CHECK: [1, 2, 3, 4]
  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
  call @printMemrefF32(%U): (memref<*xf32>) -> ()

  memref.dealloc %A : memref<4xf32>

  return
}

func.func private @mlirAsyncRuntimePrintCurrentThreadId() -> ()

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
