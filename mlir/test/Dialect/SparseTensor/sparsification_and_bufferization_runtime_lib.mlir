// RUN: mlir-opt %s --sparsification-and-bufferization --canonicalize | FileCheck %s --check-prefix=CODEGEN
// RUN: mlir-opt %s --sparsification-and-bufferization="enable-runtime-library=false" --canonicalize | FileCheck %s --check-prefix=CODEGEN
// RUN: mlir-opt %s --sparsification-and-bufferization="enable-runtime-library=true" --canonicalize | FileCheck %s --check-prefix=RUNTIME

// CODEGEN: [[$SV:#.*]] = #sparse_tensor.encoding
#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CODEGEN-LABEL: func.func @values(
// CODEGEN-SAME:      %[[POS:[^:]+]]: memref<?xindex>,
// CODEGEN-SAME:      %[[CRD:[^:]+]]: memref<?xindex>,
// CODEGEN-SAME:      %[[VAL:[^:]+]]: memref<?xf64>,
// CODEGEN-SAME:      %[[SPEC:[^:]+]]: !sparse_tensor.storage_specifier<[[$SV]]>) -> memref<?xf64> {
// CODEGEN-NEXT:    %[[SIZE:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] val_mem_sz : !sparse_tensor.storage_specifier<[[$SV]]>
// CODEGEN-NEXT:    %[[VALUES:.*]] = memref.subview %[[VAL]][0] [%[[SIZE]]] [1] : memref<?xf64> to memref<?xf64>
// CODEGEN-NEXT:    return %[[VALUES]] : memref<?xf64>
// CODEGEN-NEXT:  }

// RUNTIME-LABEL: func.func @values(
// RUNTIME-SAME:      %[[TENSOR:.*]]: !llvm.ptr) -> memref<?xf64> {
// RUNTIME-NEXT:    %[[VALUES:.*]] = call @sparseValuesF64(%[[TENSOR]]) : (!llvm.ptr) -> memref<?xf64>
// RUNTIME-NEXT:    return %[[VALUES]] : memref<?xf64>
// RUNTIME-NEXT:  }

func.func @values(%A: tensor<?xf64, #SV>) -> memref<?xf64> {
  %values = sparse_tensor.values %A : tensor<?xf64, #SV> to memref<?xf64>
  return %values : memref<?xf64>
}

// CODEGEN-LABEL: func.func @empty(
// CODEGEN-SAME:      %[[DIM:[^:]+]]: index) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<[[$SV]]>) {
// CODEGEN:         %[[C0:.*]] = arith.constant 0 : index
// CODEGEN:         %[[POS_ALLOC:.*]] = memref.alloc() : memref<16xindex>
// CODEGEN-NEXT:    %[[POS:.*]] = memref.cast %[[POS_ALLOC]] : memref<16xindex> to memref<?xindex>
// CODEGEN-NEXT:    %[[CRD_ALLOC:.*]] = memref.alloc() : memref<16xindex>
// CODEGEN-NEXT:    %[[CRD:.*]] = memref.cast %[[CRD_ALLOC]] : memref<16xindex> to memref<?xindex>
// CODEGEN-NEXT:    %[[VAL_ALLOC:.*]] = memref.alloc() : memref<16xf64>
// CODEGEN-NEXT:    %[[VAL:.*]] = memref.cast %[[VAL_ALLOC]] : memref<16xf64> to memref<?xf64>
// CODEGEN-NEXT:    %[[INIT:.*]] = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<[[$SV]]>
// CODEGEN-NEXT:    %[[SPEC:.*]] = sparse_tensor.storage_specifier.set %[[INIT]] lvl_sz at 0 with %[[DIM]] : !sparse_tensor.storage_specifier<[[$SV]]>
// CODEGEN-NEXT:    %[[POS_SIZE:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] pos_mem_sz at 0 : !sparse_tensor.storage_specifier<[[$SV]]>
// CODEGEN:         memref.store %[[C0]], %[[POS1:.*]][%[[POS_SIZE]]] : memref<?xindex>
// CODEGEN-NEXT:    %[[SPEC1:.*]] = sparse_tensor.storage_specifier.set %[[SPEC]] pos_mem_sz at 0 with %[[SIZE1:.*]] : !sparse_tensor.storage_specifier<[[$SV]]>
// CODEGEN:         memref.store %[[C0]], %[[POS2:.*]][%[[SIZE1]]] : memref<?xindex>
// CODEGEN-NEXT:    %[[SPEC2:.*]] = sparse_tensor.storage_specifier.set %[[SPEC1]] pos_mem_sz at 0 with %[[SIZE2:.*]] : !sparse_tensor.storage_specifier<[[$SV]]>
// CODEGEN-NEXT:    return %[[POS2]], %[[CRD]], %[[VAL]], %[[SPEC2]] : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<[[$SV]]>
// CODEGEN-NEXT:  }

// RUNTIME-LABEL: func.func @empty(
// RUNTIME-SAME:      %[[DIM:[^:]+]]: index) -> !llvm.ptr {
// RUNTIME-NEXT:    %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// RUNTIME-DAG:     %[[C1_I32:.*]] = arith.constant 1 : i32
// RUNTIME-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// RUNTIME-DAG:     %[[C0:.*]] = arith.constant 0 : index
// RUNTIME-DAG:     %[[COMPRESSED:.*]] = arith.constant 262144 : i64
// RUNTIME:         %[[LVL_TYPES_ALLOC:.*]] = memref.alloca() : memref<1xi64>
// RUNTIME-NEXT:    %[[LVL_TYPES:.*]] = memref.cast %[[LVL_TYPES_ALLOC]] : memref<1xi64> to memref<?xi64>
// RUNTIME-NEXT:    memref.store %[[COMPRESSED]], %[[LVL_TYPES_ALLOC]][%[[C0]]] : memref<1xi64>
// RUNTIME-NEXT:    %[[SIZES_ALLOC:.*]] = memref.alloca() : memref<1xindex>
// RUNTIME-NEXT:    %[[SIZES:.*]] = memref.cast %[[SIZES_ALLOC]] : memref<1xindex> to memref<?xindex>
// RUNTIME-NEXT:    memref.store %[[DIM]], %[[SIZES_ALLOC]][%[[C0]]] : memref<1xindex>
// RUNTIME-NEXT:    %[[MAP_ALLOC:.*]] = memref.alloca() : memref<1xindex>
// RUNTIME-NEXT:    %[[MAP:.*]] = memref.cast %[[MAP_ALLOC]] : memref<1xindex> to memref<?xindex>
// RUNTIME-NEXT:    memref.store %[[C0]], %[[MAP_ALLOC]][%[[C0]]] : memref<1xindex>
// RUNTIME-NEXT:    %[[TENSOR:.*]] = call @newSparseTensor(%[[SIZES]], %[[SIZES]], %[[LVL_TYPES]], %[[MAP]], %[[MAP]], %[[C0_I32]], %[[C0_I32]], %[[C1_I32]], %[[C0_I32]], %[[NULL]]) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
// RUNTIME-NEXT:    call @endLexInsert(%[[TENSOR]]) : (!llvm.ptr) -> ()
// RUNTIME-NEXT:    return %[[TENSOR]] : !llvm.ptr
// RUNTIME-NEXT:  }

func.func @empty(%size: index) -> tensor<?xf64, #SV> {
  %A = tensor.empty(%size) : tensor<?xf64, #SV>
  return %A : tensor<?xf64, #SV>
}

// CODEGEN-LABEL: func.func @dealloc(
// CODEGEN-SAME:      %{{[^:]+}}: memref<?xindex>,
// CODEGEN-SAME:      %{{[^:]+}}: memref<?xindex>,
// CODEGEN-SAME:      %{{[^:]+}}: memref<?xf64>,
// CODEGEN-SAME:      %{{[^:]+}}: !sparse_tensor.storage_specifier<[[$SV]]>) {
// CODEGEN-NEXT:    return
// CODEGEN-NEXT:  }

// RUNTIME-LABEL: func.func @dealloc(
// RUNTIME-SAME:      %[[TENSOR:.*]]: !llvm.ptr) {
// RUNTIME-NEXT:    call @delSparseTensor(%[[TENSOR]]) : (!llvm.ptr) -> ()
// RUNTIME-NEXT:    return
// RUNTIME-NEXT:  }

func.func @dealloc(%A: tensor<?xf64, #SV>) {
  bufferization.dealloc_tensor %A : tensor<?xf64, #SV>
  return
}
