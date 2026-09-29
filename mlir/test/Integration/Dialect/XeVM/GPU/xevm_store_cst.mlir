// RUN: mlir-opt %s \
// RUN: | mlir-opt -pass-pipeline='builtin.module(cse,func.func(gpu-async-region),xevm-attach-target,gpu.module(convert-gpu-to-llvm-spv{use-64bit-index=true},convert-xevm-to-llvm,cse))' \
// RUN: | mlir-opt -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm \
// RUN: | mlir-opt -gpu-to-llvm -reconcile-unrealized-casts -cse -gpu-module-to-binary \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @gemm attributes {gpu.container_module} {

  gpu.module @kernel {
    gpu.func @store_constant(%ptr: !llvm.ptr<1>) kernel {
      %const_val = arith.constant 42.0 : f32
      %thread_x = gpu.lane_id
      %thread_x_i64 = arith.index_cast %thread_x : index to i64
      %ptr_next_1 = llvm.getelementptr %ptr[%thread_x_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %const_val, %ptr_next_1 : f32, !llvm.ptr<1>
      gpu.return
    }
  }
  func.func @test(%src : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_0 = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_0, %src : memref<8x16xf32>, memref<8x16xf32>
    %0 = memref.extract_aligned_pointer_as_index %memref_0 : memref<8x16xf32> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %src_casted = llvm.addrspacecast %2 : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@store_constant blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%src_casted : !llvm.ptr<1>)
    %dst = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %dst, %memref_0 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_0 : memref<8x16xf32>

    return %dst : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.11 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_f32, %A[%i, %j] : memref<8x16xf32>
      }
    }
    %B = call @test(%A) : (memref<8x16xf32>) -> memref<8x16xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11.11{{.*}}]
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    memref.dealloc %A : memref<8x16xf32>
    memref.dealloc %B : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
