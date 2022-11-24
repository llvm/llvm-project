// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @invalid_new_dense(%arg0: !llvm.ptr<i8>) -> tensor<32xf32> {
  // expected-error@+1 {{'sparse_tensor.new' op result #0 must be sparse tensor of any type values, but got 'tensor<32xf32>'}}
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

func.func @invalid_pointers_dense(%arg0: tensor<128xf64>) -> memref<?xindex> {
  // expected-error@+1 {{'sparse_tensor.pointers' op operand #0 must be sparse tensor of any type values, but got 'tensor<128xf64>'}}
  %0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<128xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func.func @invalid_pointers_unranked(%arg0: tensor<*xf64>) -> memref<?xindex> {
  // expected-error@+1 {{'sparse_tensor.pointers' op operand #0 must be sparse tensor of any type values, but got 'tensor<*xf64>'}}
  %0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<*xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"], pointerBitWidth=32}>

func.func @mismatch_pointers_types(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  // expected-error@+1 {{unexpected type for pointers}}
  %0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func.func @pointers_oob(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  // expected-error@+1 {{requested pointers dimension out of bounds}}
  %0 = sparse_tensor.pointers %arg0 { dimension = 1 : index } : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func.func @invalid_indices_dense(%arg0: tensor<10x10xi32>) -> memref<?xindex> {
  // expected-error@+1 {{'sparse_tensor.indices' op operand #0 must be sparse tensor of any type values, but got 'tensor<10x10xi32>'}}
  %0 = sparse_tensor.indices %arg0 { dimension = 1 : index } : tensor<10x10xi32> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func.func @invalid_indices_unranked(%arg0: tensor<*xf64>) -> memref<?xindex> {
  // expected-error@+1 {{'sparse_tensor.indices' op operand #0 must be sparse tensor of any type values, but got 'tensor<*xf64>'}}
  %0 = sparse_tensor.indices %arg0 { dimension = 0 : index } : tensor<*xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func.func @mismatch_indices_types(%arg0: tensor<?xf64, #SparseVector>) -> memref<?xi32> {
  // expected-error@+1 {{unexpected type for indices}}
  %0 = sparse_tensor.indices %arg0 { dimension = 0 : index } : tensor<?xf64, #SparseVector> to memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func.func @indices_oob(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  // expected-error@+1 {{requested indices dimension out of bounds}}
  %0 = sparse_tensor.indices %arg0 { dimension = 1 : index } : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func.func @invalid_values_dense(%arg0: tensor<1024xf32>) -> memref<?xf32> {
  // expected-error@+1 {{'sparse_tensor.values' op operand #0 must be sparse tensor of any type values, but got 'tensor<1024xf32>'}}
  %0 = sparse_tensor.values %arg0 : tensor<1024xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func.func @mismatch_values_types(%arg0: tensor<?xf64, #SparseVector>) -> memref<?xf32> {
  // expected-error@+1 {{unexpected mismatch in element types}}
  %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

func.func @sparse_unannotated_load(%arg0: tensor<16x32xf64>) -> tensor<16x32xf64> {
  // expected-error@+1 {{'sparse_tensor.load' op operand #0 must be sparse tensor of any type values, but got 'tensor<16x32xf64>'}}
  %0 = sparse_tensor.load %arg0 : tensor<16x32xf64>
  return %0 : tensor<16x32xf64>
}

// -----

func.func @sparse_unannotated_insert(%arg0: tensor<128xf64>, %arg1: index, %arg2: f64) {
  // expected-error@+1 {{'sparse_tensor.insert' 'tensor' must be sparse tensor of any type values, but got 'tensor<128xf64>'}}
  sparse_tensor.insert %arg2 into %arg0[%arg1] : tensor<128xf64>
  return
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>

func.func @sparse_wrong_arity_insert(%arg0: tensor<128x64xf64, #CSR>, %arg1: index, %arg2: f64) {
  // expected-error@+1 {{'sparse_tensor.insert' op incorrect number of indices}}
  sparse_tensor.insert %arg2 into %arg0[%arg1] : tensor<128x64xf64, #CSR>
  return
}

// -----

func.func @sparse_push_back(%arg0: memref<?xindex>, %arg1: memref<?xf64>, %arg2: f32) -> memref<?xf64> {
  // expected-error@+1 {{'sparse_tensor.push_back' op failed to verify that value type matches element type of inBuffer}}
  %0 = sparse_tensor.push_back %arg0, %arg1, %arg2 {idx = 2 : index} : memref<?xindex>, memref<?xf64>, f32
  return %0 : memref<?xf64>
}

// -----

func.func @sparse_push_back_n(%arg0: memref<?xindex>, %arg1: memref<?xf32>, %arg2: f32) -> memref<?xf32> {
  %c0 = arith.constant 0: index
  // expected-error@+1 {{'sparse_tensor.push_back' op n must be not less than 1}}
  %0 = sparse_tensor.push_back %arg0, %arg1, %arg2, %c0 {idx = 2 : index} : memref<?xindex>, memref<?xf32>, f32, index
  return %0 : memref<?xf32>
}

// -----

func.func @sparse_unannotated_expansion(%arg0: tensor<128xf64>) {
  // expected-error@+1 {{'sparse_tensor.expand' op operand #0 must be sparse tensor of any type values, but got 'tensor<128xf64>'}}
  %values, %filled, %added, %count = sparse_tensor.expand %arg0
    : tensor<128xf64> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return
}

// -----

func.func @sparse_unannotated_compression(%arg0: memref<?xf64>,
                                          %arg1: memref<?xi1>,
                                          %arg2: memref<?xindex>,
                                          %arg3: index,
                                          %arg4: tensor<8x8xf64>,
                                          %arg5: index) {
  // expected-error@+1 {{'sparse_tensor.compress' op operand #4 must be sparse tensor of any type values, but got 'tensor<8x8xf64>'}}
  sparse_tensor.compress %arg0, %arg1, %arg2, %arg3 into %arg4[%arg5]
    : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64>
  return
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>

func.func @sparse_wrong_arity_compression(%arg0: memref<?xf64>,
                                          %arg1: memref<?xi1>,
                                          %arg2: memref<?xindex>,
                                          %arg3: index,
                                          %arg4: tensor<8x8xf64, #CSR>,
                                          %arg5: index) {
  // expected-error@+1 {{'sparse_tensor.compress' op incorrect number of indices}}
  sparse_tensor.compress %arg0, %arg1, %arg2, %arg3 into %arg4[%arg5,%arg5]
    : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64, #CSR>
  return
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func.func @sparse_new(%arg0: !llvm.ptr<i8>) {
  // expected-error@+1 {{expand_symmetry can only be used for 2D tensors}}
  %0 = sparse_tensor.new expand_symmetry %arg0 : !llvm.ptr<i8> to tensor<128xf64, #SparseVector>
  return
}

// -----

func.func @sparse_convert_unranked(%arg0: tensor<*xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{unexpected type in convert}}
  %0 = sparse_tensor.convert %arg0 : tensor<*xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

func.func @sparse_convert_rank_mismatch(%arg0: tensor<10x10xf64, #DCSR>) -> tensor<?xf64> {
  // expected-error@+1 {{unexpected conversion mismatch in rank}}
  %0 = sparse_tensor.convert %arg0 : tensor<10x10xf64, #DCSR> to tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>

func.func @sparse_convert_dim_mismatch(%arg0: tensor<10x?xf32>) -> tensor<10x10xf32, #CSR> {
  // expected-error@+1 {{unexpected conversion mismatch in dimension 1}}
  %0 = sparse_tensor.convert %arg0 : tensor<10x?xf32> to tensor<10x10xf32, #CSR>
  return %0 : tensor<10x10xf32, #CSR>
}

// -----

func.func @invalid_out_dense(%arg0: tensor<10xf64>, %arg1: !llvm.ptr<i8>) {
  // expected-error@+1 {{'sparse_tensor.out' op operand #0 must be sparse tensor of any type values, but got 'tensor<10xf64>'}}
  sparse_tensor.out %arg0, %arg1 : tensor<10xf64>, !llvm.ptr<i8>
  return
}

// -----

func.func @invalid_binary_num_args_mismatch_overlap(%arg0: f64, %arg1: f64) -> f64 {
  // expected-error@+1 {{overlap region must have exactly 2 arguments}}
  %r = sparse_tensor.binary %arg0, %arg1 : f64, f64 to f64
    overlap={
      ^bb0(%x: f64):
        sparse_tensor.yield %x : f64
    }
    left={}
    right={}
  return %r : f64
}

// -----

func.func @invalid_binary_num_args_mismatch_right(%arg0: f64, %arg1: f64) -> f64 {
  // expected-error@+1 {{right region must have exactly 1 arguments}}
  %r = sparse_tensor.binary %arg0, %arg1 : f64, f64 to f64
    overlap={}
    left={}
    right={
      ^bb0(%x: f64, %y: f64):
        sparse_tensor.yield %y : f64
    }
  return %r : f64
}

// -----

func.func @invalid_binary_argtype_mismatch(%arg0: f64, %arg1: f64) -> f64 {
  // expected-error@+1 {{overlap region argument 2 type mismatch}}
  %r = sparse_tensor.binary %arg0, %arg1 : f64, f64 to f64
    overlap={
      ^bb0(%x: f64, %y: f32):
        sparse_tensor.yield %x : f64
    }
    left=identity
    right=identity
  return %r : f64
}

// -----

func.func @invalid_binary_wrong_return_type(%arg0: f64, %arg1: f64) -> f64 {
  // expected-error@+1 {{left region yield type mismatch}}
  %0 = sparse_tensor.binary %arg0, %arg1 : f64, f64 to f64
    overlap={}
    left={
      ^bb0(%x: f64):
        %1 = arith.constant 0.0 : f32
        sparse_tensor.yield %1 : f32
    }
    right=identity
  return %0 : f64
}

// -----

func.func @invalid_binary_wrong_identity_type(%arg0: i64, %arg1: f64) -> f64 {
  // expected-error@+1 {{left=identity requires first argument to have the same type as the output}}
  %0 = sparse_tensor.binary %arg0, %arg1 : i64, f64 to f64
    overlap={}
    left=identity
    right=identity
  return %0 : f64
}

// -----

func.func @invalid_binary_wrong_yield(%arg0: f64, %arg1: f64) -> f64 {
  // expected-error@+1 {{left region must end with sparse_tensor.yield}}
  %0 = sparse_tensor.binary %arg0, %arg1 : f64, f64 to f64
    overlap={}
    left={
      ^bb0(%x: f64):
        tensor.yield %x : f64
    }
    right=identity
  return %0 : f64
}

// -----

func.func @invalid_unary_argtype_mismatch(%arg0: f64) -> f64 {
  // expected-error@+1 {{present region argument 1 type mismatch}}
  %r = sparse_tensor.unary %arg0 : f64 to f64
    present={
      ^bb0(%x: index):
        sparse_tensor.yield %x : index
    }
    absent={}
  return %r : f64
}

// -----

func.func @invalid_unary_num_args_mismatch(%arg0: f64) -> f64 {
  // expected-error@+1 {{absent region must have exactly 0 arguments}}
  %r = sparse_tensor.unary %arg0 : f64 to f64
    present={}
    absent={
      ^bb0(%x: f64):
        sparse_tensor.yield %x : f64
    }
  return %r : f64
}

// -----

func.func @invalid_unary_wrong_return_type(%arg0: f64) -> f64 {
  // expected-error@+1 {{present region yield type mismatch}}
  %0 = sparse_tensor.unary %arg0 : f64 to f64
    present={
      ^bb0(%x: f64):
        %1 = arith.constant 0.0 : f32
        sparse_tensor.yield %1 : f32
    }
    absent={}
  return %0 : f64
}

// -----

func.func @invalid_unary_wrong_yield(%arg0: f64) -> f64 {
  // expected-error@+1 {{present region must end with sparse_tensor.yield}}
  %0 = sparse_tensor.unary %arg0 : f64 to f64
    present={
      ^bb0(%x: f64):
        tensor.yield %x : f64
    }
    absent={}
  return %0 : f64
}

// -----

func.func @invalid_reduce_num_args_mismatch(%arg0: f64, %arg1: f64) -> f64 {
  %cf1 = arith.constant 1.0 : f64
  // expected-error@+1 {{reduce region must have exactly 2 arguments}}
  %r = sparse_tensor.reduce %arg0, %arg1, %cf1 : f64 {
      ^bb0(%x: f64):
        sparse_tensor.yield %x : f64
    }
  return %r : f64
}

// -----

func.func @invalid_reduce_block_arg_type_mismatch(%arg0: i64, %arg1: i64) -> i64 {
  %ci1 = arith.constant 1 : i64
  // expected-error@+1 {{reduce region argument 1 type mismatch}}
  %r = sparse_tensor.reduce %arg0, %arg1, %ci1 : i64 {
      ^bb0(%x: f64, %y: f64):
        %cst = arith.constant 2 : i64
        sparse_tensor.yield %cst : i64
    }
  return %r : i64
}

// -----

func.func @invalid_reduce_return_type_mismatch(%arg0: f64, %arg1: f64) -> f64 {
  %cf1 = arith.constant 1.0 : f64
  // expected-error@+1 {{reduce region yield type mismatch}}
  %r = sparse_tensor.reduce %arg0, %arg1, %cf1 : f64 {
      ^bb0(%x: f64, %y: f64):
        %cst = arith.constant 2 : i64
        sparse_tensor.yield %cst : i64
    }
  return %r : f64
}

// -----

func.func @invalid_reduce_wrong_yield(%arg0: f64, %arg1: f64) -> f64 {
  %cf1 = arith.constant 1.0 : f64
  // expected-error@+1 {{reduce region must end with sparse_tensor.yield}}
  %r = sparse_tensor.reduce %arg0, %arg1, %cf1 : f64 {
      ^bb0(%x: f64, %y: f64):
        %cst = arith.constant 2 : i64
        tensor.yield %cst : i64
    }
  return %r : f64
}

// -----

func.func @invalid_select_num_args_mismatch(%arg0: f64) -> f64 {
  // expected-error@+1 {{select region must have exactly 1 arguments}}
  %r = sparse_tensor.select %arg0 : f64 {
      ^bb0(%x: f64, %y: f64):
        %ret = arith.constant 1 : i1
        sparse_tensor.yield %ret : i1
    }
  return %r : f64
}

// -----

func.func @invalid_select_return_type_mismatch(%arg0: f64) -> f64 {
  // expected-error@+1 {{select region yield type mismatch}}
  %r = sparse_tensor.select %arg0 : f64 {
      ^bb0(%x: f64):
        sparse_tensor.yield %x : f64
    }
  return %r : f64
}

// -----

func.func @invalid_select_wrong_yield(%arg0: f64) -> f64 {
  // expected-error@+1 {{select region must end with sparse_tensor.yield}}
  %r = sparse_tensor.select %arg0 : f64 {
      ^bb0(%x: f64):
        tensor.yield %x : f64
    }
  return %r : f64
}

// -----

#DC = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
func.func @invalid_concat_less_inputs(%arg: tensor<9x4xf64, #DC>) -> tensor<9x4xf64, #DC> {
  // expected-error@+1 {{Need at least two tensors to concatenate.}}
  %0 = sparse_tensor.concatenate %arg {dimension = 1 : index}
       : tensor<9x4xf64, #DC> to tensor<9x4xf64, #DC>
  return %0 : tensor<9x4xf64, #DC>
}

// -----

#DC = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
func.func @invalid_concat_dim(%arg0: tensor<2x4xf64, #DC>,
                              %arg1: tensor<3x4xf64, #DC>,
                              %arg2: tensor<4x4xf64, #DC>) -> tensor<9x4xf64, #DC> {
  // expected-error@+1 {{Failed to concatentate tensors with rank=2 on dimension=4}}
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 4 : index}
       : tensor<2x4xf64, #DC>,
         tensor<3x4xf64, #DC>,
         tensor<4x4xf64, #DC> to tensor<9x4xf64, #DC>
  return %0 : tensor<9x4xf64, #DC>
}

// -----

#C = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>
#DC = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
#DCC = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed", "compressed"]}>
func.func @invalid_concat_rank_mismatch(%arg0: tensor<2xf64, #C>,
                                        %arg1: tensor<3x4xf64, #DC>,
                                        %arg2: tensor<4x4x4xf64, #DCC>) -> tensor<9x4xf64, #DC> {
  // expected-error@+1 {{The input tensor $0 has a different rank (rank=1) from the output tensor (rank=2)}}
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<2xf64, #C>,
         tensor<3x4xf64, #DC>,
         tensor<4x4x4xf64, #DCC> to tensor<9x4xf64, #DC>
  return %0 : tensor<9x4xf64, #DC>
}

// -----

#DC = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
func.func @invalid_concat_size_mismatch_dyn(%arg0: tensor<?x4xf64, #DC>,
                                            %arg1: tensor<5x4xf64, #DC>,
                                            %arg2: tensor<4x4xf64, #DC>) -> tensor<9x4xf64, #DC> {
  // expected-error@+1 {{Only statically-sized input tensors are supported.}}
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<?x4xf64, #DC>,
         tensor<5x4xf64, #DC>,
         tensor<4x4xf64, #DC> to tensor<9x4xf64, #DC>
  return %0 : tensor<9x4xf64, #DC>
}

// -----

#DC = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
func.func @invalid_concat_size_mismatch(%arg0: tensor<3x4xf64, #DC>,
                                        %arg1: tensor<5x4xf64, #DC>,
                                        %arg2: tensor<4x4xf64, #DC>) -> tensor<9x4xf64, #DC> {
  // expected-error@+1 {{The concatenation dimension of the output tensor should be the sum of}}
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<3x4xf64, #DC>,
         tensor<5x4xf64, #DC>,
         tensor<4x4xf64, #DC> to tensor<9x4xf64, #DC>
  return %0 : tensor<9x4xf64, #DC>
}

// -----

#DC = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
func.func @invalid_concat_size_mismatch(%arg0: tensor<2x4xf64, #DC>,
                                        %arg1: tensor<3x3xf64, #DC>,
                                        %arg2: tensor<4x4xf64, #DC>) -> tensor<9x4xf64, #DC> {
  // expected-error@+1 {{All dimensions (expect for the concatenating one) should be equal}}
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<2x4xf64, #DC>,
         tensor<3x3xf64, #DC>,
         tensor<4x4xf64, #DC> to tensor<9x4xf64, #DC>
  return %0 : tensor<9x4xf64, #DC>
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>) -> () {
  // expected-error@+1 {{Unmatched number of arguments in the block}}
  sparse_tensor.foreach in %arg0 : tensor<2x4xf64, #DCSR> do {
    ^bb0(%1: index, %2: index, %3: index, %v: f64) :
  }
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>) -> () {
  // expected-error@+1 {{Expecting Index type for argument at index 1}}
  sparse_tensor.foreach in %arg0 : tensor<2x4xf64, #DCSR> do {
    ^bb0(%1: index, %2: f64, %v: f64) :
  }
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>) -> () {
  // expected-error@+1 {{Unmatched element type between input tensor and block argument}}
  sparse_tensor.foreach in %arg0 : tensor<2x4xf64, #DCSR> do {
    ^bb0(%1: index, %2: index, %v: f32) :
  }
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>) -> () {
  // expected-error@+1 {{Unmatched element type between input tensor and block argument}}
  sparse_tensor.foreach in %arg0 : tensor<2x4xf64, #DCSR> do {
    ^bb0(%1: index, %2: index, %v: f32) :
  }
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>, %arg1: f32) -> () {
  // expected-error@+1 {{Mismatch in number of init arguments and results}}
  sparse_tensor.foreach in %arg0 init(%arg1) : tensor<2x4xf64, #DCSR>, f32 do {
    ^bb0(%1: index, %2: index, %v: f32, %r1 : i32) :
  }
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>, %arg1: f32) -> () {
  // expected-error@+1 {{Mismatch in types of init arguments and results}}
  %1 = sparse_tensor.foreach in %arg0 init(%arg1) : tensor<2x4xf64, #DCSR>, f32 -> i32 do {
    ^bb0(%1: index, %2: index, %v: f32, %r0 : f32) :
  }
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>, %arg1: f32) -> () {
  // expected-error@+1 {{Mismatch in types of yield values and results}}
  %1 = sparse_tensor.foreach in %arg0 init(%arg1) : tensor<2x4xf64, #DCSR>, f32 -> f32 do {
    ^bb0(%1: index, %2: index, %v: f32, %r0 : f32) :
      sparse_tensor.yield %1 : index
  }
  return
}

// -----

// TODO: a test case with empty xs doesn't work due to some parser issues.

func.func @sparse_sort_x_type( %arg0: index, %arg1: memref<?xf32>) {
  // expected-error@+1 {{operand #1 must be 1D memref of integer or index values}}
  sparse_tensor.sort %arg0, %arg1: memref<?xf32>
}

// -----

func.func @sparse_sort_dim_too_small(%arg0: memref<10xindex>) {
  %i20 = arith.constant 20 : index
  // expected-error@+1 {{xs and ys need to have a dimension >= n: 10 < 20}}
  sparse_tensor.sort %i20, %arg0 : memref<10xindex>
  return
}

// -----

func.func @sparse_sort_mismatch_x_type(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<10xi8>) {
  // expected-error@+1 {{mismatch xs element types}}
  sparse_tensor.sort %arg0, %arg1, %arg2 : memref<10xindex>, memref<10xi8>
  return
}

// -----

func.func @sparse_sort_coo_x_type( %arg0: index, %arg1: memref<?xf32>) {
  // expected-error@+1 {{operand #1 must be 1D memref of integer or index values}}
  sparse_tensor.sort_coo %arg0, %arg1: memref<?xf32>
  return
}

// -----

func.func @sparse_sort_coo_x_too_small(%arg0: memref<50xindex>) {
  %i20 = arith.constant 20 : index
  // expected-error@+1 {{Expected dimension(xy) >= n * (nx + ny) got 50 < 60}}
  sparse_tensor.sort_coo %i20, %arg0 {nx = 2 : index, ny = 1 : index} : memref<50xindex>
  return
}

// -----

func.func @sparse_sort_coo_y_too_small(%arg0: memref<60xindex>, %arg1: memref<10xf32>) {
  %i20 = arith.constant 20 : index
  // expected-error@+1 {{Expected dimension(y) >= n got 10 < 20}}
  sparse_tensor.sort_coo %i20, %arg0 jointly %arg1 {nx = 2 : index, ny = 1 : index} : memref<60xindex> jointly memref<10xf32>
  return
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>

func.func @sparse_alloc_escapes(%arg0: index) -> tensor<10x?xf64, #CSR> {
  // expected-error@+1 {{sparse tensor allocation should not escape function}}
  %0 = bufferization.alloc_tensor(%arg0) : tensor<10x?xf64, #CSR>
  return %0: tensor<10x?xf64, #CSR>
}
