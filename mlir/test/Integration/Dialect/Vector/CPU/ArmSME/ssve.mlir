// DEFINE: %{entry_point} = entry
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:  -march=aarch64 -mattr=+sve,+sme \
// DEFINE:  -e %{entry_point} -entry-point-result=i32 \
// DEFINE:  -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// NOTE: To run this test, your CPU must support SME.

// RUN: %{compile} | %{run} | FileCheck %s

// VLA memcopy in streaming mode.
func.func @streaming_kernel_copy(%src : memref<?xi64>, %dst : memref<?xi64>, %size : index) attributes {arm_streaming} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %vscale = vector.vscale
  %step = arith.muli %c2, %vscale : index
  scf.for %i = %c0 to %size step %step {
    %0 = vector.load %src[%i] : memref<?xi64>, vector<[2]xi64>
    vector.store %0, %dst[%i] : memref<?xi64>, vector<[2]xi64>
  }
  return
}

func.func @entry() -> i32 {
  %i0 = arith.constant 0: i64
  %r0 = arith.constant 0: i32
  %c0 = arith.constant 0: index
  %c4 = arith.constant 4: index
  %c32 = arith.constant 32: index

  // Set up memory.
  %a = memref.alloc()      : memref<32xi64>
  %a_copy = memref.alloc() : memref<32xi64>
  %a_data = arith.constant dense<[1 , 2,  3 , 4 , 5,  6,  7,  8,
                                  9, 10, 11, 12, 13, 14, 15, 16,
                                  17, 18, 19, 20, 21, 22, 23, 24,
                                  25, 26, 27, 28, 29, 30, 31, 32]> : vector<32xi64>
  vector.transfer_write %a_data, %a[%c0] : vector<32xi64>, memref<32xi64>

  // Call kernel.
  %0 = memref.cast %a : memref<32xi64> to memref<?xi64>
  %1 = memref.cast %a_copy : memref<32xi64> to memref<?xi64>
  call @streaming_kernel_copy(%0, %1, %c32) : (memref<?xi64>, memref<?xi64>, index) -> ()

  // Print and verify.
  //
  // CHECK:      ( 1, 2, 3, 4 )
  // CHECK-NEXT: ( 5, 6, 7, 8 )
  // CHECK-NEXT: ( 9, 10, 11, 12 )
  // CHECK-NEXT: ( 13, 14, 15, 16 )
  // CHECK-NEXT: ( 17, 18, 19, 20 )
  // CHECK-NEXT: ( 21, 22, 23, 24 )
  // CHECK-NEXT: ( 25, 26, 27, 28 )
  // CHECK-NEXT: ( 29, 30, 31, 32 )
  scf.for %i = %c0 to %c32 step %c4 {
    %cv = vector.transfer_read %a_copy[%i], %i0 : memref<32xi64>, vector<4xi64>
    vector.print %cv : vector<4xi64>
  }

  // Release resources.
  memref.dealloc %a      : memref<32xi64>
  memref.dealloc %a_copy : memref<32xi64>

  return %r0 : i32
}
