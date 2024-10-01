// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage \
// DEFINE:    -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = reduce_1d_f32
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// RUN: %{compile}

// RUN: %{run} | FileCheck %s --check-prefix=REDUCE-F32

// REDEFINE: %{entry_point} = reduce_1d_i32
// RUN: %{run} | FileCheck %s --check-prefix=REDUCE-I32

// REDEFINE: %{entry_point} = generic_reduce_1d_f32
// RUN: %{run} | FileCheck %s --check-prefix=GENERIC-F32

func.func @reduce_1d_f32() {
  // 1-D Tensor
  %N = arith.constant 1000 : index
  %c0_f32 = arith.constant 0.0 : f32

  // Allocate the input and output tensors
  %A_alloc = bufferization.alloc_tensor(%N) : tensor<?xf32>
  %C_alloc = bufferization.alloc_tensor() : tensor<f32>

  // Initialise the tensors
  %pi = arith.constant 3.1416 : f32
  %A_in = linalg.fill ins(%pi : f32) outs(%A_alloc : tensor<?xf32>) -> tensor<?xf32>
  %C_in = tensor.insert %c0_f32 into %C_alloc[] : tensor<f32>

  // Reduce
  %C_out = linalg.reduce ins(%A_in : tensor<?xf32>) outs(%C_in: tensor<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }

  // Print and verify the output
  // REDUCE-F32-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT\n"

  // REDUCE-F32-NEXT: Unranked Memref {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
  // REDUCE-F32-NEXT: [3141.6]

  %xf = tensor.cast %C_out : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  // REDUCE-F32-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT\n"

  return
}

func.func @reduce_1d_i32() {
  // 1-D Tensor
  %N = arith.constant 1000 : index
  %c0_i32 = arith.constant 0 : i32

  // Allocate the input and output tensors
  %A_alloc = bufferization.alloc_tensor(%N) : tensor<?xi32>
  %C_alloc = bufferization.alloc_tensor() : tensor<i32>

  // Initialise the tensors
  %pi = arith.constant 3 : i32
  %A_in = linalg.fill ins(%pi : i32) outs(%A_alloc : tensor<?xi32>) -> tensor<?xi32>
  %C_in = tensor.insert %c0_i32 into %C_alloc[] : tensor<i32>

  // Reduce
  %C_out = linalg.reduce ins(%A_in : tensor<?xi32>) outs(%C_in: tensor<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %0 = arith.addi %in, %init : i32
      linalg.yield %0 : i32
    }

  // Print and verify the output
  // REDUCE-I32-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT\n"

  // REDUCE-I32-NEXT: Unranked Memref {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
  // REDUCE-I32-NEXT: [3000]

  %xf = tensor.cast %C_out : tensor<i32> to tensor<*xi32>
  call @printMemrefI32(%xf) : (tensor<*xi32>) -> ()

  // REDUCE-I32-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT\n"

  return
}

func.func @generic_reduce_1d_f32() {
  // 1-D Tensor
  %N = arith.constant 1000 : index
  %c0_f32 = arith.constant 0.0 : f32

  // Allocate the input and output tensors
  %A_alloc = bufferization.alloc_tensor(%N) : tensor<?xf32>
  %C_alloc = bufferization.alloc_tensor() : tensor<f32>

  // Initialise the tensors
  %pi = arith.constant 3.1416 : f32
  %A_in = linalg.fill ins(%pi : f32) outs(%A_alloc : tensor<?xf32>) -> tensor<?xf32>
  %C_in = tensor.insert %c0_f32 into %C_alloc[] : tensor<f32>

  // Reduce
  %C_out = linalg.generic { indexing_maps = [affine_map<(d0) -> (d0)>,
                                             affine_map<(d0) -> ()>],
                            iterator_types = ["reduction"] }
    ins(%A_in : tensor<?xf32>)
    outs(%C_in : tensor<f32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<f32>

  // Print and verify the output
  // GENERIC-F32-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT\n"

  // GENERIC-F32-NEXT: Unranked Memref {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
  // GENERIC-F32-NEXT: [3141.6]

  %xf = tensor.cast %C_out : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  // GENERIC-F32-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT\n"

  return
}

module attributes {transform.with_named_sequence} {
  // A sequence that will tile and vectorise a Reduce Op
  transform.named_sequence @tile_and_vectorize_reduce(%func
    : !transform.op<"func.func"> {transform.readonly}) {

    // Step 0: Get a handle to the reduce Op
    %reduce = transform.structured.match ops{["linalg.reduce", "linalg.generic"]} in %func
      : (!transform.op<"func.func">) -> !transform.any_op

    // Step 1: Tile
    %tiled_reduce, %loops:1 = transform.structured.tile_using_for %reduce tile_sizes [[4]]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 2: Vectorize
    transform.structured.vectorize %tiled_reduce vector_sizes [[4]] : !transform.any_op

    // Step 3: Lower vector.multi_reduction
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">

    transform.yield
  }

  // A sequence that goes over all functions in tis module and applies
  // "tile_and_vectorize_reduce"
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.op<"func.func">

    transform.foreach %funcs : !transform.op<"func.func"> {
      ^bb2(%func : !transform.op<"func.func">):
        transform.include @tile_and_vectorize_reduce failures(propagate)
        (%func) : (!transform.op<"func.func">) -> ()
    }
    transform.yield
  }
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
func.func private @printMemrefI32(%ptr : tensor<*xi32>)
