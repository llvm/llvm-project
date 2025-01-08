// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:   --buffer-deallocation-simplification -canonicalize -split-input-file %s | FileCheck %s

// No ownership is assumed for ops that do not implement any interface and have
// no memref operands.

// CHECK-LABEL: func private @no_interface_no_operands(
//  CHECK-NEXT:   %[[m:.*]] = bufferization.to_memref
//  CHECK-NEXT:   %[[clone:.*]] = bufferization.clone %[[m]]
//  CHECK-NEXT:   return %[[clone]]
func.func private @no_interface_no_operands(%t : tensor<?x?x?xf16>) -> memref<?x?x?xf16> {
  %0 = bufferization.to_memref %t : tensor<?x?x?xf16> to memref<?x?x?xf16>
  return %0 : memref<?x?x?xf16>
}

// -----

// If an op does not implement any interface but has memref operands, the
// ownership of the memref results is computed from the operands.

// CHECK-LABEL: func private @no_interface(
//       CHECK:   %[[true:.*]] = arith.constant true
//       CHECK:   %[[alloc:.*]] = memref.alloc
//       CHECK:   %[[foo:.*]] = "test.forward_buffer"(%[[alloc]])
//       CHECK:   %[[r:.*]] = bufferization.dealloc (%[[alloc]] : {{.*}}) if (%[[true]]) retain (%[[foo]] : {{.*}})
//       CHECK:   return %[[foo]]
func.func private @no_interface() -> memref<5xf32> {
  %0 = memref.alloc() : memref<5xf32>
  %1 = "test.forward_buffer"(%0) : (memref<5xf32>) -> (memref<5xf32>)
  return %1 : memref<5xf32>
}

// -----

func.func @no_side_effects() {
  %0 = memref.alloc() : memref<5xf32>
  // expected-error @below{{ops with unknown memory side effects are not supported}}
  "test.unregistered_op_foo"(%0) : (memref<5xf32>) -> ()
  return
}

// -----

// Buffer deallocation should not emit any error here as the operation does not
// operate on buffers and has known memory effect (write).
func.func @no_buffer_semantics_with_write_effect(%v0: vector<9x6xf32>) {
  vector.print %v0 : vector<9x6xf32>
  return
}
