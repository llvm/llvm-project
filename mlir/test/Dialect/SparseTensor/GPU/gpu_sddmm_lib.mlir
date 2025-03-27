// RUN: mlir-opt %s --sparse-gpu-codegen="num-threads=0" | FileCheck %s

#BSR = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i floordiv 2 : dense,
    j floordiv 2 : compressed,
    i mod 2 : dense,
    j mod 2 : dense)
}>

#trait_SDDMM = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // S (in/out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "S(i,j) += spy[S(i,j)] x SUM_k A(i,k) B(k,j)"
}

// CHECK-LABEL:   func.func @SDDMM_block(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_1]] : tensor<?x?xf32> to memref<?x?xf32>
// CHECK:           %[[VAL_12:.*]] = gpu.wait async
// CHECK:           %[[VAL_13:.*]] = memref.dim %[[VAL_11]], %[[VAL_3]] : memref<?x?xf32>
// CHECK:           %[[VAL_14:.*]] = memref.dim %[[VAL_11]], %[[VAL_4]] : memref<?x?xf32>
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = gpu.alloc async {{\[}}%[[VAL_12]]] (%[[VAL_13]], %[[VAL_14]]) : memref<?x?xf32>
// CHECK:           %[[VAL_17:.*]] = gpu.memcpy async {{\[}}%[[VAL_16]]] %[[VAL_15]], %[[VAL_11]] : memref<?x?xf32>, memref<?x?xf32>
// CHECK:           %[[VAL_18:.*]] = bufferization.to_memref %[[VAL_2]] : tensor<?x?xf32> to memref<?x?xf32>
// CHECK:           %[[VAL_19:.*]] = gpu.wait async
// CHECK:           %[[VAL_20:.*]] = memref.dim %[[VAL_18]], %[[VAL_3]] : memref<?x?xf32>
// CHECK:           %[[VAL_21:.*]] = memref.dim %[[VAL_18]], %[[VAL_4]] : memref<?x?xf32>
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = gpu.alloc async {{\[}}%[[VAL_19]]] (%[[VAL_20]], %[[VAL_21]]) : memref<?x?xf32>
// CHECK:           %[[VAL_24:.*]] = gpu.memcpy async {{\[}}%[[VAL_23]]] %[[VAL_22]], %[[VAL_18]] : memref<?x?xf32>, memref<?x?xf32>
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index}
// CHECK:           %[[VAL_26:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index}
// CHECK:           %[[VAL_27:.*]] = sparse_tensor.values %[[VAL_0]]
// CHECK:           %[[VAL_28:.*]] = gpu.wait async
// CHECK:           %[[VAL_29:.*]] = memref.dim %[[VAL_25]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = gpu.alloc async {{\[}}%[[VAL_28]]] (%[[VAL_29]]) : memref<?xindex>
// CHECK:           %[[VAL_32:.*]] = gpu.memcpy async {{\[}}%[[VAL_31]]] %[[VAL_30]], %[[VAL_25]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_33:.*]] = gpu.wait async
// CHECK:           %[[VAL_34:.*]] = memref.dim %[[VAL_26]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = gpu.alloc async {{\[}}%[[VAL_33]]] (%[[VAL_34]]) : memref<?xindex>
// CHECK:           %[[VAL_37:.*]] = gpu.memcpy async {{\[}}%[[VAL_36]]] %[[VAL_35]], %[[VAL_26]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_38:.*]] = gpu.wait async
// CHECK:           %[[VAL_39:.*]] = memref.dim %[[VAL_27]], %[[VAL_3]] : memref<?xf32>
// CHECK:           %[[VAL_40:.*]], %[[VAL_41:.*]] = gpu.alloc async {{\[}}%[[VAL_38]]] (%[[VAL_39]]) : memref<?xf32>
// CHECK:           %[[VAL_42:.*]] = gpu.memcpy async {{\[}}%[[VAL_41]]] %[[VAL_40]], %[[VAL_27]] : memref<?xf32>, memref<?xf32>
// CHECK:           gpu.wait {{\[}}%[[VAL_17]], %[[VAL_24]], %[[VAL_32]], %[[VAL_37]], %[[VAL_42]]]
// CHECK:           %[[VAL_43:.*]] = gpu.wait async
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_43]]] %[[VAL_15]], %[[VAL_8]], %[[VAL_9]] : index, index into memref<?x?xf32>
// CHECK:           %[[VAL_46:.*]], %[[VAL_47:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_45]]] %[[VAL_22]], %[[VAL_9]], %[[VAL_10]] : index, index into memref<?x?xf32>
// CHECK:           %[[VAL_48:.*]] = arith.divui %[[VAL_8]], %[[VAL_5]] : index
// CHECK:           %[[VAL_49:.*]] = arith.divui %[[VAL_10]], %[[VAL_5]] : index
// CHECK:           %[[VAL_50:.*]] = arith.divui %[[VAL_7]], %[[VAL_6]] : index
// CHECK:           %[[VAL_51:.*]], %[[VAL_52:.*]] = gpu.create_bsr async {{\[}}%[[VAL_47]]] %[[VAL_48]], %[[VAL_49]], %[[VAL_50]], %[[VAL_5]], %[[VAL_5]], %[[VAL_30]], %[[VAL_35]], %[[VAL_40]] : memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:           %[[VAL_53:.*]], %[[VAL_54:.*]] = gpu.sddmm_buffer_size async {{\[}}%[[VAL_52]]] %[[VAL_44]], %[[VAL_46]], %[[VAL_51]] into f32
// CHECK:           %[[VAL_55:.*]], %[[VAL_56:.*]] = gpu.alloc async {{\[}}%[[VAL_54]]] (%[[VAL_53]]) : memref<?xi8>
// CHECK:           %[[VAL_57:.*]] = gpu.sddmm async {{\[}}%[[VAL_56]]] %[[VAL_44]], %[[VAL_46]], %[[VAL_51]], %[[VAL_55]] : memref<?xi8> into f32
// CHECK:           %[[VAL_58:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_57]]] %[[VAL_44]]
// CHECK:           %[[VAL_59:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_58]]] %[[VAL_46]]
// CHECK:           %[[VAL_60:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_59]]] %[[VAL_51]]
// CHECK:           %[[VAL_61:.*]] = gpu.dealloc async {{\[}}%[[VAL_60]]] %[[VAL_55]] : memref<?xi8>
// CHECK:           %[[VAL_62:.*]] = gpu.dealloc async {{\[}}%[[VAL_61]]] %[[VAL_15]] : memref<?x?xf32>
// CHECK:           %[[VAL_63:.*]] = gpu.dealloc async {{\[}}%[[VAL_62]]] %[[VAL_22]] : memref<?x?xf32>
// CHECK:           %[[VAL_64:.*]] = gpu.dealloc async {{\[}}%[[VAL_63]]] %[[VAL_30]] : memref<?xindex>
// CHECK:           %[[VAL_65:.*]] = gpu.dealloc async {{\[}}%[[VAL_64]]] %[[VAL_35]] : memref<?xindex>
// CHECK:           %[[VAL_66:.*]] = gpu.memcpy async {{\[}}%[[VAL_65]]] %[[VAL_27]], %[[VAL_40]] : memref<?xf32>, memref<?xf32>
// CHECK:           %[[VAL_67:.*]] = gpu.dealloc async {{\[}}%[[VAL_66]]] %[[VAL_40]] : memref<?xf32>
// CHECK:           gpu.wait {{\[}}%[[VAL_67]]]
// CHECK:           %[[VAL_68:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_68]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @SDDMM_block(%args: tensor<?x?xf32, #BSR>,
                       %arga: tensor<?x?xf32>,
                       %argb: tensor<?x?xf32>) -> tensor<?x?xf32, #BSR> {
  %result = linalg.generic #trait_SDDMM
      ins(%arga, %argb: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%args: tensor<?x?xf32, #BSR>) {
        ^bb(%a: f32, %b: f32, %s: f32):
           %f0 = arith.constant 0.0 : f32
           %u = sparse_tensor.unary %s : f32 to f32
             present={
                ^bb0(%p: f32):
                  %mul = arith.mulf %a, %b : f32
                  sparse_tensor.yield %mul : f32
             }
             absent={}
           %r = sparse_tensor.reduce %s, %u, %f0 : f32 {
              ^bb0(%p: f32, %q: f32):
                %add = arith.addf %p, %q : f32
                sparse_tensor.yield %add : f32
            }
           linalg.yield %r : f32
      } -> tensor<?x?xf32, #BSR>
  return %result : tensor<?x?xf32, #BSR>
}
