// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

// Unrolling scalable reduction dim is not supported - bail out

// expected-error@below {{greedy pattern application failed}}
func.func @masked_extract_contract2_scalable_reduction_dim(%arg0: vector<[2]x[3]xf32>,
                                    %arg1: vector<[3]xf32>,
                                    %arg2: vector<[2]xf32>,
                                    %m: vector<[2]x[3]xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.contract #matvec_trait %arg0, %arg1, %arg2
          : vector<[2]x[3]xf32>, vector<[3]xf32> into vector<[2]xf32> } : vector<[2]x[3]xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %f = transform.structured.match ops{["func.func"]} in %module_op 
    : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
  } : !transform.any_op
}
