// RUN: mlir-opt %s --test-transform-dialect-interpreter --verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @_reduce_leading_trailing(%entry: !transform.any_op {transform.readonly})
      -> (!transform.any_op) {
    %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>

    transform.match.structured %entry : !transform.any_op {
    ^bb0(%struct: !transform.any_op):
      transform.match.structured.dim %struct[all] {parallel} : !transform.any_op
      transform.match.structured.input %struct[all] {projected_permutation} : !transform.any_op
      transform.match.structured.init %struct[all] {permutation} : !transform.any_op
      %ni = transform.match.structured.num_inits %struct : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %ni, %c1 : !transform.param<i64>
    }
    transform.yield %entry : !transform.any_op
  }

  transform.named_sequence @fill_reduce_leading_trailing(%entry: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op,
          !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    %c4 = transform.param.constant 4 : i64 -> !transform.param<i64>

    %rk, %dms, %bw, %operand_o, %init_v, %trailing_o = transform.match.structured failures(propagate) %entry 
        : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
                                  !transform.any_op, !transform.any_value, !transform.any_op) {
    ^bb0(%struct: !transform.any_op):
      %rank = transform.match.structured.rank %struct : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi ge %rank, %c2 : !transform.param<i64>
      transform.match.param.cmpi le %rank, %c4 : !transform.param<i64>
      
      transform.match.structured.dim %struct[-1] {reduction} : !transform.any_op
      transform.match.structured.dim %struct[except(-1)] {parallel} : !transform.any_op
      %dims = transform.match.structured.dim %struct[all] : (!transform.any_op) -> !transform.param<i64>

      %n_inputs = transform.match.structured.num_inputs %struct : (!transform.any_op) -> !transform.param<i64>
      %n_outputs = transform.match.structured.num_inits %struct : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_inputs, %c1 : !transform.param<i64>
      transform.match.param.cmpi eq %n_outputs, %c1 : !transform.param<i64>

      transform.match.structured.input %struct[0] {projected_permutation} : !transform.any_op
      transform.match.structured.init %struct[0] {projected_permutation} : !transform.any_op
      %init = transform.match.structured.init %struct[0] : (!transform.any_op) -> !transform.any_value
      
      // This danse is necessary to create an empty handle if there is no single
      // user without failing the entire match
      %trailing_optional = transform.sequence %struct : (!transform.any_op) -> !transform.any_op failures(suppress) {
      ^bb0(%struct_inner: !transform.any_op):
        %result = transform.match.structured failures(propagate) %struct_inner : (!transform.any_op) -> !transform.any_op {
        ^bb0(%struct_inner_inner: !transform.any_op):
          %result_inner = transform.match.structured.result %struct_inner_inner[0] {single} : (!transform.any_op) -> !transform.any_op
          %trailing = transform.include @_reduce_leading_trailing failures(propagate) (%result_inner) : (!transform.any_op) -> !transform.any_op
          transform.match.structured.yield %trailing : !transform.any_op
        }
        transform.yield %result: !transform.any_op
      }

      // Suppress errors as a way to implement optionality. We cannot suppress them in
      // the include because it keeps matching after "get_defining_op" fails, which
      // breaks the single-op precondition of the following ops. We don't want to
      // propagate that failure though.
      //
      // Additionally, we cannot put the sequence inside the call because its first
      // operand must be an operation handle (the verifier asserts!) and there is
      // no such handle available there.
      //
      // TODO: extend the structured matching to gracefully handle empty handles
      // or provide the suppress-errors-but-stop failure mode for includes to
      // implement optionality.
      %operand_optional = transform.sequence %struct : (!transform.any_op) -> !transform.any_op failures(suppress) {
      ^bb0(%struct_inner: !transform.any_op):
        %operand3 = transform.match.structured failures(propagate) %struct_inner : (!transform.any_op) -> !transform.any_op {
        ^bb1(%struct_inner_inner: !transform.any_op):
          %operand = transform.match.structured.input %struct_inner_inner[0] : (!transform.any_op) -> !transform.any_op
          %operand2 = transform.include @_reduce_leading_trailing failures(propagate) (%operand) : (!transform.any_op) -> !transform.any_op
          transform.match.structured.yield %operand2 : !transform.any_op
        }
        transform.yield %operand3 : !transform.any_op
      }

      %bitwidth = transform.match.structured.elemental_bitwidth %init : (!transform.any_value) -> !transform.param<i64>

      transform.match.structured.body %struct { reduction_position = 0 } : !transform.any_op
      transform.match.structured.yield %rank, %dims, %bitwidth, %operand_optional, %init, %trailing_optional
        : !transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
          !transform.any_op, !transform.any_value, !transform.any_op
    }

    %init_o = transform.get_defining_op %init_v : (!transform.any_value) -> !transform.any_op
    transform.match.operation_name %init_o ["linalg.fill"] : !transform.any_op    

    transform.yield %operand_o, %init_o, %entry, %trailing_o, %rk, %dms, %bw
        : !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op,
          !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
  }

  transform.named_sequence @print_reduce_leading_trailing(
      %leading: !transform.any_op {transform.readonly},
      %fill: !transform.any_op {transform.readonly},
      %reduction: !transform.any_op {transform.readonly},
      %trailing: !transform.any_op {transform.readonly},
      %rank: !transform.param<i64> {transform.readonly},
      %dims: !transform.param<i64> {transform.readonly},
      %bitwidth: !transform.param<i64> {transform.readonly}) {
    transform.test_print_remark_at_operand %leading, "leading" : !transform.any_op
    transform.test_print_remark_at_operand %fill, "fill" : !transform.any_op
    transform.test_print_remark_at_operand %reduction, "reduction" : !transform.any_op
    transform.test_print_remark_at_operand %trailing, "trailing" : !transform.any_op
    transform.test_print_param %rank, "rank" at %reduction : !transform.param<i64>, !transform.any_op
    transform.test_print_param %dims, "dimensions" at %reduction : !transform.param<i64>, !transform.any_op
    transform.test_print_param %bitwidth, "bitwidth" at %reduction : !transform.param<i64>, !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb(%root: !transform.any_op):
    foreach_match in %root
      @fill_reduce_leading_trailing -> @print_reduce_leading_trailing
      : (!transform.any_op) -> !transform.any_op
  }
}

!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @eltwise_reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  %2 = tensor.empty() : !in_tensor_t
  // expected-remark @below {{leading}}
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  // expected-remark @below {{reduction}}
  // expected-remark @below {{rank 2}}
  // expected-remark @below {{dimensions 8 : i64, 64 : i64}}
  // expected-remark @below {{bitwidth 32 : i64}}
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  return %6 : !out_tensor_t
}

func.func @reduce_eltwise(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) -> !out_tensor_t
  // expected-remark @below {{reduction}}
  // expected-remark @below {{rank 2}}
  // expected-remark @below {{dimensions 8 : i64, 64 : i64}}
  // expected-remark @below {{bitwidth 32 : i64}}
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %6 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{trailing}}
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%5 : !out_tensor_t) outs(%6 : !out_tensor_t) {  
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t
  return %7 : !out_tensor_t
}

func.func @eltwise_reduce_eltwise(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  %2 = tensor.empty() : !in_tensor_t
  // expected-remark @below {{leading}}
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  // expected-remark @below {{reduction}}
  // expected-remark @below {{rank 2}}
  // expected-remark @below {{dimensions 8 : i64, 64 : i64}}
  // expected-remark @below {{bitwidth 32 : i64}}
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %7 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{trailing}}
  %8 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%6 : !out_tensor_t) outs(%7 : !out_tensor_t) {  
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t


  return %8 : !out_tensor_t
}

func.func @eltwise_reduce_eltwise_swapped(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %2 = tensor.empty() : !in_tensor_t
  // expected-remark @below {{leading}}
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  // expected-remark @below {{reduction}}
  // expected-remark @below {{rank 2}}
  // expected-remark @below {{dimensions 8 : i64, 64 : i64}}
  // expected-remark @below {{bitwidth 32 : i64}}
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %7 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{trailing}}
  %8 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%6 : !out_tensor_t) outs(%7 : !out_tensor_t) {  
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t


  return %8 : !out_tensor_t
}

func.func @reduction_with_extra_op_in_func(%arg0: tensor<8x479xf32>, %arg1: tensor<32x32xf32>) -> (tensor<8xf32>, tensor<32xf32>) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  // expected-remark @below {{fill}}
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  // expected-remark @below {{reduction}}
  // expected-remark @below {{rank 2}}
  // expected-remark @below {{dimensions 8 : i64, 479 : i64}}
  // expected-remark @below {{bitwidth 32 : i64}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg0 : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>

  %empty2 = tensor.empty() : tensor<32xf32>
  %fill2 = linalg.fill ins(%cst : f32) outs(%empty2 : tensor<32xf32>) -> tensor<32xf32>
  return %result, %fill2 : tensor<8xf32>, tensor<32xf32>
}
