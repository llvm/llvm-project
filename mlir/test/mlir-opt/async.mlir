// Check if mlir marks the corresponding function with required coroutine attribute.
//
// RUN:   mlir-opt %s -async-to-async-runtime                                  \
// RUN:               -async-runtime-ref-counting                              \
// RUN:               -async-runtime-ref-counting-opt                          \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-linalg-to-loops                                 \
// RUN:               -convert-scf-to-std                                      \
// RUN:               -convert-linalg-to-llvm                                  \
// RUN:               -convert-memref-to-llvm                                  \
// RUN:               -convert-arith-to-llvm                                   \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | FileCheck %s

// CHECK: llvm.func @async_execute_fn{{.*}}attributes{{.*}}"coroutine.presplit", "0"
// CHECK: llvm.func @async_execute_fn_0{{.*}}attributes{{.*}}"coroutine.presplit", "0"
// CHECK: llvm.func @async_execute_fn_1{{.*}}attributes{{.*}}"coroutine.presplit", "0"

func @main() {
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
  linalg.fill(%c0, %A) : f32, memref<4xf32>

  %U = memref.cast %A :  memref<4xf32> to memref<*xf32>
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  memref.store %c1, %A[%i0]: memref<4xf32>
  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  %outer = async.execute {
    memref.store %c2, %A[%i1]: memref<4xf32>
    call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    call @print_memref_f32(%U): (memref<*xf32>) -> ()

    // No op async region to create a token for testing async dependency.
    %noop = async.execute {
      call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
      async.yield
    }

    %inner = async.execute [%noop] {
      memref.store %c3, %A[%i2]: memref<4xf32>
      call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
      call @print_memref_f32(%U): (memref<*xf32>) -> ()

      async.yield
    }
    async.await %inner : !async.token

    memref.store %c4, %A[%i3]: memref<4xf32>
    call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    call @print_memref_f32(%U): (memref<*xf32>) -> ()

    async.yield
  }
  async.await %outer : !async.token

  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  memref.dealloc %A : memref<4xf32>

  return
}

func private @mlirAsyncRuntimePrintCurrentThreadId() -> ()

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
