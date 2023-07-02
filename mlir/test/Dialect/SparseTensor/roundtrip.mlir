// RUN: mlir-opt %s -split-input-file | mlir-opt -split-input-file | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_new(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = sparse_tensor.new %[[A]] : !llvm.ptr<i8> to tensor<128xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<128xf64, #{{.*}}>
func.func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<128xf64, #SparseVector>
  return %0 : tensor<128xf64, #SparseVector>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"], posWidth=32, crdWidth=32}>

// CHECK-LABEL: func @sparse_pack(
// CHECK-SAME: %[[D:.*]]: tensor<6xf64>,
// CHECK-SAME: %[[P:.*]]: tensor<2xi32>,
// CHECK-SAME: %[[I:.*]]: tensor<6x1xi32>)
//       CHECK: %[[R:.*]] = sparse_tensor.pack %[[D]], %[[P]], %[[I]]
//       CHECK: return %[[R]] : tensor<100xf64, #{{.*}}>
func.func @sparse_pack(%data: tensor<6xf64>, %pos: tensor<2xi32>, %index: tensor<6x1xi32>)
                            -> tensor<100xf64, #SparseVector> {
  %0 = sparse_tensor.pack %data, %pos, %index : tensor<6xf64>, tensor<2xi32>, tensor<6x1xi32>
                                             to tensor<100xf64, #SparseVector>
  return %0 : tensor<100xf64, #SparseVector>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"], crdWidth=32}>
// CHECK-LABEL: func @sparse_unpack(
//  CHECK-SAME: %[[T:.*]]: tensor<100xf64, #
//  CHECK-SAME: %[[OD:.*]]: tensor<6xf64>
//  CHECK-SAME: %[[OP:.*]]: tensor<2xindex>
//  CHECK-SAME: %[[OI:.*]]: tensor<6x1xi32>
//       CHECK: %[[D:.*]], %[[P:.*]]:2, %[[DL:.*]], %[[PL:.*]]:2 = sparse_tensor.unpack %[[T]]
//       CHECK: return %[[D]], %[[P]]#0, %[[P]]#1
func.func @sparse_unpack(%sp : tensor<100xf64, #SparseVector>,
                         %od : tensor<6xf64>,
                         %op : tensor<2xindex>,
                         %oi : tensor<6x1xi32>)
                       -> (tensor<6xf64>, tensor<2xindex>, tensor<6x1xi32>) {
  %rd, %rp, %ri, %vl, %pl, %cl = sparse_tensor.unpack %sp : tensor<100xf64, #SparseVector>
                  outs(%od, %op, %oi : tensor<6xf64>, tensor<2xindex>, tensor<6x1xi32>)
                  -> tensor<6xf64>, (tensor<2xindex>, tensor<6x1xi32>), index, (index, index)
  return %rd, %rp, %ri : tensor<6xf64>, tensor<2xindex>, tensor<6x1xi32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_dealloc(
// CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>
//       CHECK: bufferization.dealloc_tensor %[[A]] : tensor<128xf64, #{{.*}}>
//       CHECK: return
func.func @sparse_dealloc(%arg0: tensor<128xf64, #SparseVector>) {
  bufferization.dealloc_tensor %arg0 : tensor<128xf64, #SparseVector>
  return
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_convert_1d_to_sparse(
// CHECK-SAME: %[[A:.*]]: tensor<64xf32>)
//       CHECK: %[[T:.*]] = sparse_tensor.convert %[[A]] : tensor<64xf32> to tensor<64xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<64xf32, #{{.*}}>
func.func @sparse_convert_1d_to_sparse(%arg0: tensor<64xf32>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// -----

#SparseTensor = #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense", "compressed" ] }>

// CHECK-LABEL: func @sparse_convert_3d_from_sparse(
// CHECK-SAME: %[[A:.*]]: tensor<8x8x8xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.convert %[[A]] : tensor<8x8x8xf64, #{{.*}}> to tensor<8x8x8xf64>
//       CHECK: return %[[T]] : tensor<8x8x8xf64>
func.func @sparse_convert_3d_from_sparse(%arg0: tensor<8x8x8xf64, #SparseTensor>) -> tensor<8x8x8xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<8x8x8xf64, #SparseTensor> to tensor<8x8x8xf64>
  return %0 : tensor<8x8x8xf64>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_positions(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.positions %[[A]] {level = 0 : index} : tensor<128xf64, #{{.*}}> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func.func @sparse_positions(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %0 = sparse_tensor.positions %arg0 {level = 0 : index} : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#COO = #sparse_tensor.encoding<{lvlTypes = ["compressed-nu", "singleton"]}>

// CHECK-LABEL: func @sparse_indices_buffer(
//  CHECK-SAME: %[[A:.*]]: tensor<?x?xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.coordinates_buffer %[[A]] : tensor<?x?xf64, #{{.*}}> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func.func @sparse_indices_buffer(%arg0: tensor<?x?xf64, #COO>) -> memref<?xindex> {
  %0 = sparse_tensor.coordinates_buffer %arg0 : tensor<?x?xf64, #COO> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_indices(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.coordinates %[[A]] {level = 0 : index} : tensor<128xf64, #{{.*}}> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func.func @sparse_indices(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %0 = sparse_tensor.coordinates %arg0 {level = 0 : index} : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_values(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.values %[[A]] : tensor<128xf64, #{{.*}}> to memref<?xf64>
//       CHECK: return %[[T]] : memref<?xf64>
func.func @sparse_values(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<128xf64, #SparseVector> to memref<?xf64>
  return %0 : memref<?xf64>
}

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimSlices = [ (1, 4, 1), (1, 4, 2) ]
}>

// CHECK-LABEL: func @sparse_slice_offset(
//  CHECK-SAME: %[[A:.*]]: tensor<2x8xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.slice.offset %[[A]] at 1 : tensor<2x8xf64, #{{.*}}>
//       CHECK: return %[[T]] : index
func.func @sparse_slice_offset(%arg0: tensor<2x8xf64, #CSR_SLICE>) -> index {
  %0 = sparse_tensor.slice.offset %arg0 at 1 : tensor<2x8xf64, #CSR_SLICE>
  return %0 : index
}

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimSlices = [ (1, 4, 1), (1, 4, 2) ]
}>

// CHECK-LABEL: func @sparse_slice_stride(
//  CHECK-SAME: %[[A:.*]]: tensor<2x8xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.slice.stride %[[A]] at 1 : tensor<2x8xf64, #{{.*}}>
//       CHECK: return %[[T]] : index
func.func @sparse_slice_stride(%arg0: tensor<2x8xf64, #CSR_SLICE>) -> index {
  %0 = sparse_tensor.slice.stride %arg0 at 1 : tensor<2x8xf64, #CSR_SLICE>
  return %0 : index
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_metadata_init(
//       CHECK: %[[T:.*]] = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#{{.*}}>
//       CHECK: return %[[T]] : !sparse_tensor.storage_specifier<#{{.*}}>
func.func @sparse_metadata_init() -> !sparse_tensor.storage_specifier<#SparseVector> {
  %0 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#SparseVector>
  return %0 : !sparse_tensor.storage_specifier<#SparseVector>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>
#SparseVector_Slice = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"],
  dimSlices = [ (?, ?, ?) ]
}>

// CHECK-LABEL: func @sparse_metadata_init(
//  CHECK-SAME: %[[A:.*]]: !sparse_tensor.storage_specifier<#{{.*}}>
//       CHECK: %[[T:.*]] = sparse_tensor.storage_specifier.init with %[[A]] :
//       CHECK: return %[[T]] : !sparse_tensor.storage_specifier<#{{.*}}>
func.func @sparse_metadata_init(%src : !sparse_tensor.storage_specifier<#SparseVector>)
                                    -> !sparse_tensor.storage_specifier<#SparseVector_Slice> {
  %0 = sparse_tensor.storage_specifier.init with %src : from !sparse_tensor.storage_specifier<#SparseVector>
                                                          to !sparse_tensor.storage_specifier<#SparseVector_Slice>
  return %0 : !sparse_tensor.storage_specifier<#SparseVector_Slice>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_get_md(
//  CHECK-SAME: %[[A:.*]]: !sparse_tensor.storage_specifier<#{{.*}}>
//       CHECK: %[[T:.*]] = sparse_tensor.storage_specifier.get %[[A]] lvl_sz at 0
//       CHECK: return %[[T]] : index
func.func @sparse_get_md(%arg0: !sparse_tensor.storage_specifier<#SparseVector>) -> index {
  %0 = sparse_tensor.storage_specifier.get %arg0 lvl_sz at 0
       : !sparse_tensor.storage_specifier<#SparseVector>
  return %0 : index
}

// -----

#SparseVector_Slice = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"],
  dimSlices = [ (?, ?, ?) ]
}>

// CHECK-LABEL: func @sparse_get_md(
//  CHECK-SAME: %[[A:.*]]: !sparse_tensor.storage_specifier<#{{.*}}>
//       CHECK: %[[T:.*]] = sparse_tensor.storage_specifier.get %[[A]] dim_offset at 0
//       CHECK: return %[[T]] : index
func.func @sparse_get_md(%arg0: !sparse_tensor.storage_specifier<#SparseVector_Slice>) -> index {
  %0 = sparse_tensor.storage_specifier.get %arg0 dim_offset at 0
       : !sparse_tensor.storage_specifier<#SparseVector_Slice>
  return %0 : index
}

// -----

#SparseVector = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"],
  dimSlices = [ (?, ?, ?) ]
}>

// CHECK-LABEL: func @sparse_get_md(
//  CHECK-SAME: %[[A:.*]]: !sparse_tensor.storage_specifier<#{{.*}}>
//       CHECK: %[[T:.*]] = sparse_tensor.storage_specifier.get %[[A]] dim_stride at 0
//       CHECK: return %[[T]] : index
func.func @sparse_get_md(%arg0: !sparse_tensor.storage_specifier<#SparseVector>) -> index {
  %0 = sparse_tensor.storage_specifier.get %arg0 dim_stride at 0
       : !sparse_tensor.storage_specifier<#SparseVector>
  return %0 : index
}


// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_set_md(
//  CHECK-SAME: %[[A:.*]]: !sparse_tensor.storage_specifier<#{{.*}}>,
//  CHECK-SAME: %[[I:.*]]: index)
//       CHECK: %[[T:.*]] = sparse_tensor.storage_specifier.set %[[A]] lvl_sz at 0 with %[[I]]
//       CHECK: return %[[T]] : !sparse_tensor.storage_specifier<#{{.*}}>
func.func @sparse_set_md(%arg0: !sparse_tensor.storage_specifier<#SparseVector>, %arg1: index)
          -> !sparse_tensor.storage_specifier<#SparseVector> {
  %0 = sparse_tensor.storage_specifier.set %arg0 lvl_sz at 0 with %arg1
       : !sparse_tensor.storage_specifier<#SparseVector>
  return %0 : !sparse_tensor.storage_specifier<#SparseVector>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_noe(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.number_of_entries %[[A]] : tensor<128xf64, #{{.*}}>
//       CHECK: return %[[T]] : index
func.func @sparse_noe(%arg0: tensor<128xf64, #SparseVector>) -> index {
  %0 = sparse_tensor.number_of_entries %arg0 : tensor<128xf64, #SparseVector>
  return %0 : index
}

// -----

#DenseMatrix = #sparse_tensor.encoding<{lvlTypes = ["dense","dense"]}>

// CHECK-LABEL: func @sparse_load(
//  CHECK-SAME: %[[A:.*]]: tensor<16x32xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.load %[[A]] : tensor<16x32xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<16x32xf64, #{{.*}}>
func.func @sparse_load(%arg0: tensor<16x32xf64, #DenseMatrix>) -> tensor<16x32xf64, #DenseMatrix> {
  %0 = sparse_tensor.load %arg0 : tensor<16x32xf64, #DenseMatrix>
  return %0 : tensor<16x32xf64, #DenseMatrix>
}

// -----

#DenseMatrix = #sparse_tensor.encoding<{lvlTypes = ["dense","dense"]}>

// CHECK-LABEL: func @sparse_load_ins(
//  CHECK-SAME: %[[A:.*]]: tensor<16x32xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.load %[[A]] hasInserts : tensor<16x32xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<16x32xf64, #{{.*}}>
func.func @sparse_load_ins(%arg0: tensor<16x32xf64, #DenseMatrix>) -> tensor<16x32xf64, #DenseMatrix> {
  %0 = sparse_tensor.load %arg0 hasInserts : tensor<16x32xf64, #DenseMatrix>
  return %0 : tensor<16x32xf64, #DenseMatrix>
}

// -----

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

// CHECK-LABEL: func @sparse_insert(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #sparse_tensor.encoding<{{.*}}>>,
//  CHECK-SAME: %[[B:.*]]: index,
//  CHECK-SAME: %[[C:.*]]: f64)
//       CHECK: %[[T:.*]] = sparse_tensor.insert %[[C]] into %[[A]][%[[B]]] : tensor<128xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<128xf64, #{{.*}}>
func.func @sparse_insert(%arg0: tensor<128xf64, #SparseVector>, %arg1: index, %arg2: f64) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.insert %arg2 into %arg0[%arg1] : tensor<128xf64, #SparseVector>
  return %0 : tensor<128xf64, #SparseVector>
}

// -----

// CHECK-LABEL: func @sparse_push_back(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> (memref<?xf64>, index) {
//       CHECK: %[[D:.*]] = sparse_tensor.push_back %[[A]], %[[B]], %[[C]] : index, memref<?xf64>, f64
//       CHECK: return %[[D]]
func.func @sparse_push_back(%arg0: index, %arg1: memref<?xf64>, %arg2: f64) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back %arg0, %arg1, %arg2 : index, memref<?xf64>, f64
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

// CHECK-LABEL: func @sparse_push_back_inbound(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> (memref<?xf64>, index) {
//       CHECK: %[[D:.*]] = sparse_tensor.push_back inbounds %[[A]], %[[B]], %[[C]] : index, memref<?xf64>, f64
//       CHECK: return %[[D]]
func.func @sparse_push_back_inbound(%arg0: index, %arg1: memref<?xf64>, %arg2: f64) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back inbounds %arg0, %arg1, %arg2 : index, memref<?xf64>, f64
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

// CHECK-LABEL: func @sparse_push_back_n(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64,
//  CHECK-SAME: %[[D:.*]]: index) -> (memref<?xf64>, index) {
//       CHECK: %[[E:.*]] = sparse_tensor.push_back %[[A]], %[[B]], %[[C]], %[[D]] : index, memref<?xf64>, f64, index
//       CHECK: return %[[E]]
func.func @sparse_push_back_n(%arg0: index, %arg1: memref<?xf64>, %arg2: f64, %arg3: index) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back %arg0, %arg1, %arg2, %arg3 : index, memref<?xf64>, f64, index
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_expansion(
//  CHECK-SAME: %[[A:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>)
//       CHECK: %{{.*}}, %{{.*}}, %{{.*}}, %[[T:.*]] = sparse_tensor.expand %[[A]]
//       CHECK: return %[[T]] : index
func.func @sparse_expansion(%tensor: tensor<8x8xf64, #SparseMatrix>) -> index {
  %values, %filled, %added, %count = sparse_tensor.expand %tensor
    : tensor<8x8xf64, #SparseMatrix> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %count : index
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_compression(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xf64>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi1>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*3]]: index
//  CHECK-SAME: %[[A4:.*4]]: tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>,
//  CHECK-SAME: %[[A5:.*5]]: index)
//       CHECK: %[[T:.*]] = sparse_tensor.compress %[[A0]], %[[A1]], %[[A2]], %[[A3]] into %[[A4]][%[[A5]]
//       CHECK: return %[[T]] : tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>
func.func @sparse_compression(%values: memref<?xf64>,
                              %filled: memref<?xi1>,
                              %added: memref<?xindex>,
                              %count: index,
			      %tensor: tensor<8x8xf64, #SparseMatrix>,
			      %index: index) -> tensor<8x8xf64, #SparseMatrix> {
  %0 = sparse_tensor.compress %values, %filled, %added, %count into %tensor[%index]
    : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64, #SparseMatrix>
  return %0 : tensor<8x8xf64, #SparseMatrix>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_out(
//  CHECK-SAME: %[[A:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{{.*}}>>,
//  CHECK-SAME: %[[B:.*]]: !llvm.ptr<i8>)
//       CHECK: sparse_tensor.out %[[A]], %[[B]] : tensor<?x?xf64, #sparse_tensor.encoding<{{.*}}>>, !llvm.ptr<i8>
//       CHECK: return
func.func @sparse_out(%arg0: tensor<?x?xf64, #SparseMatrix>, %arg1: !llvm.ptr<i8>) {
  sparse_tensor.out %arg0, %arg1 : tensor<?x?xf64, #SparseMatrix>, !llvm.ptr<i8>
  return
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_binary(
//  CHECK-SAME:   %[[A:.*]]: f64, %[[B:.*]]: i64) -> f64 {
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:   %[[C1:.*]] = sparse_tensor.binary %[[A]], %[[B]] : f64, i64 to f64
//       CHECK:     overlap = {
//       CHECK:       ^bb0(%[[A1:.*]]: f64, %[[B1:.*]]: i64):
//       CHECK:         sparse_tensor.yield %[[A1]] : f64
//       CHECK:     }
//       CHECK:     left = identity
//       CHECK:     right = {
//       CHECK:       ^bb0(%[[A2:.*]]: i64):
//       CHECK:         sparse_tensor.yield %[[Z]] : f64
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_binary(%arg0: f64, %arg1: i64) -> f64 {
  %cf0 = arith.constant 0.0 : f64
  %r = sparse_tensor.binary %arg0, %arg1 : f64, i64 to f64
    overlap={
      ^bb0(%x: f64, %y: i64):
        sparse_tensor.yield %x : f64
    }
    left=identity
    right={
      ^bb0(%y: i64):
        sparse_tensor.yield %cf0 : f64
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_unary(
//  CHECK-SAME:   %[[A:.*]]: f64) -> f64 {
//       CHECK:   %[[C1:.*]] = sparse_tensor.unary %[[A]] : f64 to f64
//       CHECK:     present = {
//       CHECK:       ^bb0(%[[A1:.*]]: f64):
//       CHECK:         sparse_tensor.yield %[[A1]] : f64
//       CHECK:     }
//       CHECK:     absent = {
//       CHECK:       %[[R:.*]] = arith.constant -1.000000e+00 : f64
//       CHECK:       sparse_tensor.yield %[[R]] : f64
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_unary(%arg0: f64) -> f64 {
  %r = sparse_tensor.unary %arg0 : f64 to f64
    present={
      ^bb0(%x: f64):
        sparse_tensor.yield %x : f64
    } absent={
      ^bb0:
        %cf1 = arith.constant -1.0 : f64
        sparse_tensor.yield %cf1 : f64
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_unary(
//  CHECK-SAME:   %[[A:.*]]: f64) -> i64 {
//       CHECK:   %[[C1:.*]] = sparse_tensor.unary %[[A]] : f64 to i64
//       CHECK:     present = {
//       CHECK:       ^bb0(%[[A1:.*]]: f64):
//       CHECK:         %[[R:.*]] = arith.fptosi %[[A1]] : f64 to i64
//       CHECK:         sparse_tensor.yield %[[R]] : i64
//       CHECK:     }
//       CHECK:     absent = {
//       CHECK:     }
//       CHECK:   return %[[C1]] : i64
//       CHECK: }
func.func @sparse_unary(%arg0: f64) -> i64 {
  %r = sparse_tensor.unary %arg0 : f64 to i64
    present={
      ^bb0(%x: f64):
        %ret = arith.fptosi %x : f64 to i64
        sparse_tensor.yield %ret : i64
    }
    absent={}
  return %r : i64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_reduce_2d_to_1d(
//  CHECK-SAME:   %[[A:.*]]: f64, %[[B:.*]]: f64) -> f64 {
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:   %[[C1:.*]] = sparse_tensor.reduce %[[A]], %[[B]], %[[Z]] : f64 {
//       CHECK:       ^bb0(%[[A1:.*]]: f64, %[[B1:.*]]: f64):
//       CHECK:         sparse_tensor.yield %[[A1]] : f64
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_reduce_2d_to_1d(%arg0: f64, %arg1: f64) -> f64 {
  %cf0 = arith.constant 0.0 : f64
  %r = sparse_tensor.reduce %arg0, %arg1, %cf0 : f64 {
      ^bb0(%x: f64, %y: f64):
        sparse_tensor.yield %x : f64
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_select(
//  CHECK-SAME:   %[[A:.*]]: f64) -> f64 {
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:   %[[C1:.*]] = sparse_tensor.select %[[A]] : f64 {
//       CHECK:       ^bb0(%[[A1:.*]]: f64):
//       CHECK:         %[[B1:.*]] = arith.cmpf ogt, %[[A1]], %[[Z]] : f64
//       CHECK:         sparse_tensor.yield %[[B1]] : i1
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_select(%arg0: f64) -> f64 {
  %cf0 = arith.constant 0.0 : f64
  %r = sparse_tensor.select %arg0 : f64 {
      ^bb0(%x: f64):
        %cmp = arith.cmpf "ogt", %x, %cf0 : f64
        sparse_tensor.yield %cmp : i1
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @concat_sparse_sparse(
//  CHECK-SAME:   %[[A0:.*]]: tensor<2x4xf64
//  CHECK-SAME:   %[[A1:.*]]: tensor<3x4xf64
//  CHECK-SAME:   %[[A2:.*]]: tensor<4x4xf64
//       CHECK:   %[[TMP0:.*]] = sparse_tensor.concatenate %[[A0]], %[[A1]], %[[A2]] {dimension = 0 : index} :
//  CHECK-SAME:   tensor<2x4xf64
//  CHECK-SAME:   tensor<3x4xf64
//  CHECK-SAME:   tensor<4x4xf64
//  CHECK-SAME:   tensor<9x4xf64
//       CHECK:   return %[[TMP0]] : tensor<9x4xf64
func.func @concat_sparse_sparse(%arg0: tensor<2x4xf64, #SparseMatrix>,
                                %arg1: tensor<3x4xf64, #SparseMatrix>,
                                %arg2: tensor<4x4xf64, #SparseMatrix>) -> tensor<9x4xf64, #SparseMatrix> {
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<2x4xf64, #SparseMatrix>,
         tensor<3x4xf64, #SparseMatrix>,
         tensor<4x4xf64, #SparseMatrix> to tensor<9x4xf64, #SparseMatrix>
  return %0 : tensor<9x4xf64, #SparseMatrix>
}

// -----

#DCSR = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_tensor_foreach(
//  CHECK-SAME: %[[A0:.*]]: tensor<2x4xf64
//       CHECK: sparse_tensor.foreach in %[[A0]] :
//       CHECK:  ^bb0(%arg1: index, %arg2: index, %arg3: f64):
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>) -> () {
  sparse_tensor.foreach in %arg0 : tensor<2x4xf64, #DCSR> do {
    ^bb0(%1: index, %2: index, %v: f64) :
  }
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_tensor_foreach(
//  CHECK-SAME:   %[[A0:.*]]: tensor<2x4xf64, #sparse_tensor.encoding<{{{.*}}}>>,
//  CHECK-SAME:   %[[A1:.*]]: f32
//  CHECK-NEXT:   %[[RET:.*]] = sparse_tensor.foreach in %[[A0]] init(%[[A1]])
//  CHECK-NEXT:    ^bb0(%[[TMP_1:.*]]: index, %[[TMP_2:.*]]: index, %[[TMP_v:.*]]: f64, %[[TMP_r:.*]]: f32)
//       CHECK:      sparse_tensor.yield %[[TMP_r]] : f32
//       CHECK:  }
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>, %arg1: f32) -> () {
  %ret = sparse_tensor.foreach in %arg0 init(%arg1): tensor<2x4xf64, #DCSR>, f32 -> f32
  do {
    ^bb0(%1: index, %2: index, %v: f64, %r: f32) :
      sparse_tensor.yield %r : f32
  }
  return
}

// -----

// CHECK-LABEL: func @sparse_sort_1d0v(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xindex>)
//       CHECK: sparse_tensor.sort hybrid_quick_sort %[[A]], %[[B]] : memref<?xindex>
//       CHECK: return %[[B]]
func.func @sparse_sort_1d0v(%arg0: index, %arg1: memref<?xindex>) -> (memref<?xindex>) {
  sparse_tensor.sort hybrid_quick_sort %arg0, %arg1 : memref<?xindex>
  return %arg1 : memref<?xindex>
}

// -----

// CHECK-LABEL: func @sparse_sort_1d2v(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<20xindex>,
//  CHECK-SAME: %[[C:.*]]: memref<10xindex>,
//  CHECK-SAME: %[[D:.*]]: memref<?xf32>)
//       CHECK: sparse_tensor.sort hybrid_quick_sort %[[A]], %[[B]] jointly %[[C]], %[[D]] : memref<20xindex> jointly memref<10xindex>, memref<?xf32>
//       CHECK: return %[[B]], %[[C]], %[[D]]
func.func @sparse_sort_1d2v(%arg0: index, %arg1: memref<20xindex>, %arg2: memref<10xindex>, %arg3: memref<?xf32>) -> (memref<20xindex>, memref<10xindex>, memref<?xf32>) {
  sparse_tensor.sort hybrid_quick_sort %arg0, %arg1 jointly %arg2, %arg3 : memref<20xindex> jointly memref<10xindex>, memref<?xf32>
  return %arg1, %arg2, %arg3 : memref<20xindex>, memref<10xindex>, memref<?xf32>
}

// -----

// CHECK-LABEL: func @sparse_sort_2d1v(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<10xi8>,
//  CHECK-SAME: %[[C:.*]]: memref<20xi8>,
//  CHECK-SAME: %[[D:.*]]: memref<10xf64>)
//       CHECK: sparse_tensor.sort hybrid_quick_sort %[[A]], %[[B]], %[[C]] jointly %[[D]] : memref<10xi8>, memref<20xi8> jointly memref<10xf64>
//       CHECK: return %[[B]], %[[C]], %[[D]]
func.func @sparse_sort_2d1v(%arg0: index, %arg1: memref<10xi8>, %arg2: memref<20xi8>, %arg3: memref<10xf64>) -> (memref<10xi8>, memref<20xi8>, memref<10xf64>) {
  sparse_tensor.sort hybrid_quick_sort %arg0, %arg1, %arg2 jointly %arg3 : memref<10xi8>, memref<20xi8> jointly memref<10xf64>
  return %arg1, %arg2, %arg3 : memref<10xi8>, memref<20xi8>, memref<10xf64>
}

// -----

// CHECK-LABEL: func @sparse_sort_stable(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<10xi8>,
//  CHECK-SAME: %[[C:.*]]: memref<20xi8>,
//  CHECK-SAME: %[[D:.*]]: memref<10xf64>)
//       CHECK: sparse_tensor.sort insertion_sort_stable %[[A]], %[[B]], %[[C]] jointly %[[D]] : memref<10xi8>, memref<20xi8> jointly memref<10xf64>
//       CHECK: return %[[B]], %[[C]], %[[D]]
func.func @sparse_sort_stable(%arg0: index, %arg1: memref<10xi8>, %arg2: memref<20xi8>, %arg3: memref<10xf64>) -> (memref<10xi8>, memref<20xi8>, memref<10xf64>) {
  sparse_tensor.sort insertion_sort_stable %arg0, %arg1, %arg2 jointly %arg3 : memref<10xi8>, memref<20xi8> jointly memref<10xf64>
  return %arg1, %arg2, %arg3 : memref<10xi8>, memref<20xi8>, memref<10xf64>
}

// -----

// CHECK-LABEL: func @sparse_sort_coo(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xindex>)
//       CHECK: sparse_tensor.sort_coo hybrid_quick_sort %[[A]], %[[B]] {nx = 2 : index, ny = 1 : index} : memref<?xindex>
//       CHECK: return %[[B]]
func.func @sparse_sort_coo(%arg0: index, %arg1: memref<?xindex>) -> (memref<?xindex>) {
  sparse_tensor.sort_coo hybrid_quick_sort %arg0, %arg1 {nx = 2 : index, ny = 1 : index}: memref<?xindex>
  return %arg1 : memref<?xindex>
}

// -----

// CHECK-LABEL: func @sparse_sort_coo_stable(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[C:.*]]: memref<?xf32>)
//       CHECK: sparse_tensor.sort_coo insertion_sort_stable %[[A]], %[[B]] jointly %[[C]] {nx = 2 : index, ny = 1 : index}
//       CHECK: return %[[B]], %[[C]]
func.func @sparse_sort_coo_stable(%arg0: index, %arg1: memref<?xi64>, %arg2: memref<?xf32>) -> (memref<?xi64>, memref<?xf32>) {
  sparse_tensor.sort_coo insertion_sort_stable %arg0, %arg1 jointly %arg2 {nx = 2 : index, ny = 1 : index}: memref<?xi64> jointly memref<?xf32>
  return %arg1, %arg2 : memref<?xi64>, memref<?xf32>
}
