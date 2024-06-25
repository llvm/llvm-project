// RUN: mlir-opt -finalize-memref-to-llvm -llvm-request-c-wrappers -convert-func-to-llvm -reconcile-unrealized-casts %s | FileCheck %s
// RUN: mlir-opt -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts %s | FileCheck %s --check-prefix=EMIT_C_ATTRIBUTE

// This tests the default memref calling convention and the emission of C
// wrappers. We don't need to separate runs because the wrapper-emission
// version subsumes the calling convention and only adds new functions, that we
// can also file-check in the same run.

// An external function is transformed into the glue around calling an interface function.
// CHECK-LABEL: @external
// CHECK: %[[ALLOC0:.*]]: !llvm.ptr, %[[ALIGN0:.*]]: !llvm.ptr, %[[OFFSET0:.*]]: i64, %[[SIZE00:.*]]: i64, %[[SIZE01:.*]]: i64, %[[STRIDE00:.*]]: i64, %[[STRIDE01:.*]]: i64,
// CHECK: %[[ALLOC1:.*]]: !llvm.ptr, %[[ALIGN1:.*]]: !llvm.ptr, %[[OFFSET1:.*]]: i64)
func.func private @external(%arg0: memref<?x?xf32>, %arg1: memref<f32>)
  // Populate the descriptor for arg0.
  // CHECK: %[[DESC00:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC01:.*]] = llvm.insertvalue %arg0, %[[DESC00]][0]
  // CHECK: %[[DESC02:.*]] = llvm.insertvalue %arg1, %[[DESC01]][1]
  // CHECK: %[[DESC03:.*]] = llvm.insertvalue %arg2, %[[DESC02]][2]
  // CHECK: %[[DESC04:.*]] = llvm.insertvalue %arg3, %[[DESC03]][3, 0]
  // CHECK: %[[DESC05:.*]] = llvm.insertvalue %arg5, %[[DESC04]][4, 0]
  // CHECK: %[[DESC06:.*]] = llvm.insertvalue %arg4, %[[DESC05]][3, 1]
  // CHECK: %[[DESC07:.*]] = llvm.insertvalue %arg6, %[[DESC06]][4, 1]

  // Allocate on stack and store to comply with C calling convention.
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[DESC0_ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.store %[[DESC07]], %[[DESC0_ALLOCA]]

  // Populate the descriptor for arg1.
  // CHECK: %[[DESC10:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[DESC11:.*]] = llvm.insertvalue %arg7, %[[DESC10]][0] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[DESC12:.*]] = llvm.insertvalue %arg8, %[[DESC11]][1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[DESC13:.*]] = llvm.insertvalue %arg9, %[[DESC12]][2] : !llvm.struct<(ptr, ptr, i64)>

  // Allocate on stack and store to comply with C calling convention.
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[DESC1_ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm.struct<(ptr, ptr, i64)>
  // CHECK: llvm.store %[[DESC13]], %[[DESC1_ALLOCA]]

  // Call the interface function.
  // CHECK: llvm.call @_mlir_ciface_external

// Verify that an interface function is emitted.
// CHECK-LABEL: llvm.func @_mlir_ciface_external
// CHECK: (!llvm.ptr, !llvm.ptr)

// Verify that the return value is not affected.
// CHECK-LABEL: @returner
// CHECK: -> !llvm.struct<(struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, struct<(ptr, ptr, i64)>)>
func.func private @returner() -> (memref<?x?xf32>, memref<f32>)

// CHECK-LABEL: @caller
func.func @caller() {
  %0:2 = call @returner() : () -> (memref<?x?xf32>, memref<f32>)
  // Extract individual values from the descriptor for the first memref.
  // CHECK: %[[ALLOC0:.*]] = llvm.extractvalue %[[DESC0:.*]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[ALIGN0:.*]] = llvm.extractvalue %[[DESC0]][1]
  // CHECK: %[[OFFSET0:.*]] = llvm.extractvalue %[[DESC0]][2]
  // CHECK: %[[SIZE00:.*]] = llvm.extractvalue %[[DESC0]][3, 0]
  // CHECK: %[[SIZE01:.*]] = llvm.extractvalue %[[DESC0]][3, 1]
  // CHECK: %[[STRIDE00:.*]] = llvm.extractvalue %[[DESC0]][4, 0]
  // CHECK: %[[STRIDE01:.*]] = llvm.extractvalue %[[DESC0]][4, 1]

  // Extract individual values from the descriptor for the second memref.
  // CHECK: %[[ALLOC1:.*]] = llvm.extractvalue %[[DESC1:.*]][0] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[ALIGN1:.*]] = llvm.extractvalue %[[DESC1]][1]
  // CHECK: %[[OFFSET1:.*]] = llvm.extractvalue %[[DESC1]][2]

  // Forward the values to the call.
  // CHECK: llvm.call @external(%[[ALLOC0]], %[[ALIGN0]], %[[OFFSET0]], %[[SIZE00]], %[[SIZE01]], %[[STRIDE00]], %[[STRIDE01]], %[[ALLOC1]], %[[ALIGN1]], %[[OFFSET1]]) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64) -> ()
  call @external(%0#0, %0#1) : (memref<?x?xf32>, memref<f32>) -> ()
  return
}

// CHECK-LABEL: @callee
// EMIT_C_ATTRIBUTE-LABEL: @callee
func.func @callee(%arg0: memref<?xf32>, %arg1: index) {
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  return
}

// Verify that an interface function is emitted.
// CHECK-LABEL: @_mlir_ciface_callee
// CHECK: %[[ARG0:.*]]: !llvm.ptr
  // Load the memref descriptor pointer.
  // CHECK: %[[DESC:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

  // Extract individual components of the descriptor.
  // CHECK: %[[ALLOC:.*]] = llvm.extractvalue %[[DESC]][0]
  // CHECK: %[[ALIGN:.*]] = llvm.extractvalue %[[DESC]][1]
  // CHECK: %[[OFFSET:.*]] = llvm.extractvalue %[[DESC]][2]
  // CHECK: %[[SIZE:.*]] = llvm.extractvalue %[[DESC]][3, 0]
  // CHECK: %[[STRIDE:.*]] = llvm.extractvalue %[[DESC]][4, 0]

  // Forward the descriptor components to the call.
  // CHECK: llvm.call @callee(%[[ALLOC]], %[[ALIGN]], %[[OFFSET]], %[[SIZE]], %[[STRIDE]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> ()

//   EMIT_C_ATTRIBUTE-NOT: @mlir_ciface_callee

// CHECK-LABEL: @other_callee
// EMIT_C_ATTRIBUTE-LABEL: @other_callee
func.func @other_callee(%arg0: memref<?xf32>, %arg1: index) attributes { llvm.emit_c_interface } {
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  return
}

// CHECK: @_mlir_ciface_other_callee
// CHECK:   llvm.call @other_callee

// EMIT_C_ATTRIBUTE: @_mlir_ciface_other_callee
// EMIT_C_ATTRIBUTE:   llvm.call @other_callee

//===========================================================================//
// Calling convention on returning unranked memrefs.
//===========================================================================//

// CHECK-LABEL: llvm.func @return_var_memref_caller
func.func @return_var_memref_caller(%arg0: memref<4x3xf32>) {
  // CHECK: %[[CALL_RES:.*]] = llvm.call @return_var_memref
  %0 = call @return_var_memref(%arg0) : (memref<4x3xf32>) -> memref<*xf32>

  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2 : index)
  // These sizes may depend on the data layout, not matching specific values.
  // CHECK: %[[IDX_SIZE:.*]] = llvm.mlir.constant

  // CHECK: %[[PTR_SIZE:.*]] = llvm.mlir.constant
  // CHECK: %[[DOUBLE_PTR_SIZE:.*]] = llvm.mul %[[TWO]], %[[PTR_SIZE]]
  // CHECK: %[[RANK:.*]] = llvm.extractvalue %[[CALL_RES]][0] : !llvm.struct<(i64, ptr)>
  // CHECK: %[[DOUBLE_RANK:.*]] = llvm.mul %[[TWO]], %[[RANK]]
  // CHECK: %[[DOUBLE_RANK_INC:.*]] = llvm.add %[[DOUBLE_RANK]], %[[ONE]]
  // CHECK: %[[TABLES_SIZE:.*]] = llvm.mul %[[DOUBLE_RANK_INC]], %[[IDX_SIZE]]
  // CHECK: %[[ALLOC_SIZE:.*]] = llvm.add %[[DOUBLE_PTR_SIZE]], %[[TABLES_SIZE]]
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOC_SIZE]] x i8
  // CHECK: %[[SOURCE:.*]] = llvm.extractvalue %[[CALL_RES]][1]
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[SOURCE]], %[[ALLOC_SIZE]]) <{isVolatile = false}>
  // CHECK: llvm.call @free(%[[SOURCE]])
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
  // CHECK: %[[RANK:.*]] = llvm.extractvalue %[[CALL_RES]][0] : !llvm.struct<(i64, ptr)>
  // CHECK: %[[DESC_1:.*]] = llvm.insertvalue %[[RANK]], %[[DESC]][0]
  // CHECK: llvm.insertvalue %[[ALLOCA]], %[[DESC_1]][1]
  return
}

// CHECK-LABEL: llvm.func @return_var_memref
func.func @return_var_memref(%arg0: memref<4x3xf32>) -> memref<*xf32> attributes { llvm.emit_c_interface } {
  // Match the construction of the unranked descriptor.
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca
  // CHECK: %[[RANK:.*]] = llvm.mlir.constant(2 : index)
  // CHECK: %[[DESC_0:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
  // CHECK: %[[DESC_1:.*]] = llvm.insertvalue %[[RANK]], %[[DESC_0]][0]
  // CHECK: %[[DESC_2:.*]] = llvm.insertvalue %[[ALLOCA]], %[[DESC_1]][1]
  %0 = memref.cast %arg0: memref<4x3xf32> to memref<*xf32>

  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2 : index)
  // These sizes may depend on the data layout, not matching specific values.
  // CHECK: %[[IDX_SIZE:.*]] = llvm.mlir.constant

  // CHECK: %[[PTR_SIZE:.*]] = llvm.mlir.constant
  // CHECK: %[[DOUBLE_PTR_SIZE:.*]] = llvm.mul %[[TWO]], %[[PTR_SIZE]]
  // CHECK: %[[RANK_EXTR:.*]] = llvm.extractvalue %[[DESC_2]][0]
  // CHECK: %[[DOUBLE_RANK:.*]] = llvm.mul %[[TWO]], %[[RANK_EXTR]]
  // CHECK: %[[DOUBLE_RANK_INC:.*]] = llvm.add %[[DOUBLE_RANK]], %[[ONE]]
  // CHECK: %[[TABLES_SIZE:.*]] = llvm.mul %[[DOUBLE_RANK_INC]], %[[IDX_SIZE]]
  // CHECK: %[[ALLOC_SIZE:.*]] = llvm.add %[[DOUBLE_PTR_SIZE]], %[[TABLES_SIZE]]
  // CHECK: %[[ALLOCATED:.*]] = llvm.call @malloc(%[[ALLOC_SIZE]])
  // CHECK: %[[ALLOCA_EXTRACTED:.*]] = llvm.extractvalue %[[DESC_2]][1]
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCATED]], %[[ALLOCA_EXTRACTED]], %[[ALLOC_SIZE]]) <{isVolatile = false}>
  // CHECK: %[[NEW_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
  // CHECK: %[[RANK_EXTRACTED:.*]] = llvm.extractvalue %[[DESC_2]][0]
  // CHECK: %[[NEW_DESC_1:.*]] = llvm.insertvalue %[[RANK_EXTRACTED]], %[[NEW_DESC]][0]
  // CHECK: %[[NEW_DESC_2:.*]] = llvm.insertvalue %[[ALLOCATED]], %[[NEW_DESC_1]][1]
  // CHECK: llvm.return %[[NEW_DESC_2]]
  return %0 : memref<*xf32>
}

// Check that the result memref is passed as parameter
// CHECK-LABEL: @_mlir_ciface_return_var_memref
// CHECK-SAME: (%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr)

// CHECK-LABEL: llvm.func @return_two_var_memref_caller
func.func @return_two_var_memref_caller(%arg0: memref<4x3xf32>) {
  // Only check that we create two different descriptors using different
  // memory, and deallocate both sources. The size computation is same as for
  // the single result.
  // CHECK: %[[CALL_RES:.*]] = llvm.call @return_two_var_memref
  // CHECK: %[[RES_1:.*]] = llvm.extractvalue %[[CALL_RES]][0]
  // CHECK: %[[RES_2:.*]] = llvm.extractvalue %[[CALL_RES]][1]
  %0:2 = call @return_two_var_memref(%arg0) : (memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>)

  // CHECK: %[[ALLOCA_1:.*]] = llvm.alloca %{{.*}} x i8
  // CHECK: %[[SOURCE_1:.*]] = llvm.extractvalue %[[RES_1:.*]][1] : ![[DESC_TYPE:.*>]]
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA_1]], %[[SOURCE_1]], %{{.*}}) <{isVolatile = false}>
  // CHECK: llvm.call @free(%[[SOURCE_1]])
  // CHECK: %[[DESC_1:.*]] = llvm.mlir.undef : ![[DESC_TYPE]]
  // CHECK: %[[DESC_11:.*]] = llvm.insertvalue %{{.*}}, %[[DESC_1]][0]
  // CHECK: llvm.insertvalue %[[ALLOCA_1]], %[[DESC_11]][1]

  // CHECK: %[[ALLOCA_2:.*]] = llvm.alloca %{{.*}} x i8
  // CHECK: %[[SOURCE_2:.*]] = llvm.extractvalue %[[RES_2:.*]][1]
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA_2]], %[[SOURCE_2]], %{{.*}}) <{isVolatile = false}>
  // CHECK: llvm.call @free(%[[SOURCE_2]])
  // CHECK: %[[DESC_2:.*]] = llvm.mlir.undef : ![[DESC_TYPE]]
  // CHECK: %[[DESC_21:.*]] = llvm.insertvalue %{{.*}}, %[[DESC_2]][0]
  // CHECK: llvm.insertvalue %[[ALLOCA_2]], %[[DESC_21]][1]
  return
}

// CHECK-LABEL: llvm.func @return_two_var_memref
func.func @return_two_var_memref(%arg0: memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>) attributes { llvm.emit_c_interface } {
  // Match the construction of the unranked descriptor.
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca
  // CHECK: %[[DESC_0:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
  // CHECK: %[[DESC_1:.*]] = llvm.insertvalue %{{.*}}, %[[DESC_0]][0]
  // CHECK: %[[DESC_2:.*]] = llvm.insertvalue %[[ALLOCA]], %[[DESC_1]][1]
  %0 = memref.cast %arg0 : memref<4x3xf32> to memref<*xf32>

  // Only check that we allocate the memory for each operand of the "return"
  // separately, even if both operands are the same value. The calling
  // convention requires the caller to free them and the caller cannot know
  // whether they are the same value or not.
  // CHECK: %[[ALLOCATED_1:.*]] = llvm.call @malloc(%{{.*}})
  // CHECK: %[[ALLOCA_EXTRACTED:.*]] = llvm.extractvalue %[[DESC_2]][1]
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCATED_1]], %[[ALLOCA_EXTRACTED]], %{{.*}}) <{isVolatile = false}>
  // CHECK: %[[RES_1:.*]] = llvm.mlir.undef
  // CHECK: %[[RES_11:.*]] = llvm.insertvalue %{{.*}}, %[[RES_1]][0]
  // CHECK: %[[RES_12:.*]] = llvm.insertvalue %[[ALLOCATED_1]], %[[RES_11]][1]

  // CHECK: %[[ALLOCATED_2:.*]] = llvm.call @malloc(%{{.*}})
  // CHECK: %[[ALLOCA_EXTRACTED:.*]] = llvm.extractvalue %[[DESC_2]][1]
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCATED_2]], %[[ALLOCA_EXTRACTED]], %{{.*}}) <{isVolatile = false}>
  // CHECK: %[[RES_2:.*]] = llvm.mlir.undef
  // CHECK: %[[RES_21:.*]] = llvm.insertvalue %{{.*}}, %[[RES_2]][0]
  // CHECK: %[[RES_22:.*]] = llvm.insertvalue %[[ALLOCATED_2]], %[[RES_21]][1]

  // CHECK: %[[RESULTS:.*]] = llvm.mlir.undef : !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)>
  // CHECK: %[[RESULTS_1:.*]] = llvm.insertvalue %[[RES_12]], %[[RESULTS]]
  // CHECK: %[[RESULTS_2:.*]] = llvm.insertvalue %[[RES_22]], %[[RESULTS_1]]
  // CHECK: llvm.return %[[RESULTS_2]]
  return %0, %0 : memref<*xf32>, memref<*xf32>
}

// Check that the result memrefs are passed as parameter
// CHECK-LABEL: @_mlir_ciface_return_two_var_memref
// CHECK-SAME: (%{{.*}}: !llvm.ptr,
// CHECK-SAME: %{{.*}}: !llvm.ptr)

// CHECK-LABEL: llvm.func @bare_ptr_calling_conv(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME: -> !llvm.ptr
func.func @bare_ptr_calling_conv(%arg0: memref<4x3xf32>, %arg1 : index, %arg2 : index, %arg3 : f32)
     -> (memref<4x3xf32>) attributes { llvm.bareptr } {
  // CHECK: %[[UNDEF_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[INSERT_ALLOCPTR:.*]] = llvm.insertvalue %[[ARG0]], %[[UNDEF_DESC]][0]
  // CHECK: %[[INSERT_ALIGNEDPTR:.*]] = llvm.insertvalue %[[ARG0]], %[[INSERT_ALLOCPTR]][1]
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[INSERT_OFFSET:.*]] = llvm.insertvalue %[[C0]], %[[INSERT_ALIGNEDPTR]][2]
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[INSERT_DIM0:.*]] = llvm.insertvalue %[[C4]], %[[INSERT_OFFSET]][3, 0]
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[INSERT_STRIDE0:.*]] = llvm.insertvalue %[[C3]], %[[INSERT_DIM0]][4, 0]
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[INSERT_DIM1:.*]] = llvm.insertvalue %[[C3]], %[[INSERT_STRIDE0]][3, 1]
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[INSERT_STRIDE1:.*]] = llvm.insertvalue %[[C1]], %[[INSERT_DIM1]][4, 1]

  // CHECK: %[[ALIGNEDPTR:.*]] = llvm.extractvalue %[[INSERT_STRIDE1]][1]
  // CHECK: %[[STOREPTR:.*]] = llvm.getelementptr %[[ALIGNEDPTR]]
  // CHECK: llvm.store %{{.*}}, %[[STOREPTR]]
  memref.store %arg3, %arg0[%arg1, %arg2] : memref<4x3xf32>

  // CHECK: %[[EXTRACT_MEMREF:.*]] = llvm.extractvalue %[[INSERT_STRIDE1]][0]
  // CHECK: llvm.return %[[EXTRACT_MEMREF]]
  return %arg0 : memref<4x3xf32>
}

// CHECK-LABEL: llvm.func @bare_ptr_calling_conv_multiresult(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME: -> !llvm.struct<(f32, ptr)>
func.func @bare_ptr_calling_conv_multiresult(%arg0: memref<4x3xf32>, %arg1 : index, %arg2 : index, %arg3 : f32)
     -> (f32, memref<4x3xf32>) attributes { llvm.bareptr } {
  // CHECK: %[[UNDEF_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[INSERT_ALLOCPTR:.*]] = llvm.insertvalue %[[ARG0]], %[[UNDEF_DESC]][0]
  // CHECK: %[[INSERT_ALIGNEDPTR:.*]] = llvm.insertvalue %[[ARG0]], %[[INSERT_ALLOCPTR]][1]
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[INSERT_OFFSET:.*]] = llvm.insertvalue %[[C0]], %[[INSERT_ALIGNEDPTR]][2]
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[INSERT_DIM0:.*]] = llvm.insertvalue %[[C4]], %[[INSERT_OFFSET]][3, 0]
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[INSERT_STRIDE0:.*]] = llvm.insertvalue %[[C3]], %[[INSERT_DIM0]][4, 0]
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[INSERT_DIM1:.*]] = llvm.insertvalue %[[C3]], %[[INSERT_STRIDE0]][3, 1]
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[INSERT_STRIDE1:.*]] = llvm.insertvalue %[[C1]], %[[INSERT_DIM1]][4, 1]

  // CHECK: %[[ALIGNEDPTR:.*]] = llvm.extractvalue %[[INSERT_STRIDE1]][1]
  // CHECK: %[[STOREPTR:.*]] = llvm.getelementptr %[[ALIGNEDPTR]]
  // CHECK: llvm.store %{{.*}}, %[[STOREPTR]]
  memref.store %arg3, %arg0[%arg1, %arg2] : memref<4x3xf32>

  // CHECK: %[[ALIGNEDPTR0:.*]] = llvm.extractvalue %[[INSERT_STRIDE1]][1]
  // CHECK: %[[LOADPTR:.*]] = llvm.getelementptr %[[ALIGNEDPTR0]]
  // CHECK: %[[RETURN0:.*]] = llvm.load %[[LOADPTR]]
  %0 = memref.load %arg0[%arg1, %arg2] : memref<4x3xf32>

  // CHECK: %[[EXTRACT_MEMREF:.*]] = llvm.extractvalue %[[INSERT_STRIDE1]][0]
  // CHECK: %[[RETURN_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(f32, ptr)>
  // CHECK: %[[INSERT_RETURN0:.*]] = llvm.insertvalue %[[RETURN0]], %[[RETURN_DESC]][0]
  // CHECK: %[[INSERT_RETURN1:.*]] = llvm.insertvalue %[[EXTRACT_MEMREF]], %[[INSERT_RETURN0]][1]
  // CHECK: llvm.return %[[INSERT_RETURN1]]
  return %0, %arg0 : f32, memref<4x3xf32>
}
