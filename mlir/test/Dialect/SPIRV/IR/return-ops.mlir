// RUN: mlir-opt %s -remove-dead-values="canonicalize=0" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -remove-dead-values="canonicalize=1" -split-input-file | FileCheck %s --check-prefix=CHECK-CANONICALIZE

// Make sure that the return value op is considered as a return-like op and
// remains live.

// CHECK-LABEL: @preserve_return_value
// CHECK-SAME:    (%[[ARG0:.*]]: vector<2xi32>, %[[ARG1:.*]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    %[[BITCAST0:.*]] = spirv.Bitcast %[[ARG1]] : vector<2xi32> to vector<2xf32>
// CHECK-NEXT:    %[[BITCAST1:.*]] = spirv.Bitcast %[[BITCAST0]] : vector<2xf32> to vector<2xi32>
// CHECK-NEXT:    spirv.ReturnValue %[[BITCAST1]] : vector<2xi32>
// CHECK-CANONICALIZE-LABEL: @preserve_return_value
// CHECK-CANONICALIZE-SAME:    (%[[ARG0:.*]]: vector<2xi32>, %[[ARG1:.*]]: vector<2xi32>) -> vector<2xi32>
// CHECK-CANONICALIZE-NEXT:    spirv.ReturnValue %[[ARG1]] : vector<2xi32>

spirv.func @preserve_return_value(%arg0: vector<2xi32>, %arg1: vector<2xi32>) -> vector<2xi32> "None" {
  %0 = spirv.Bitcast %arg0 : vector<2xi32> to vector<2xf32>
  %1 = spirv.Bitcast %arg1 : vector<2xi32> to vector<2xf32>
  %2 = spirv.Bitcast %1 : vector<2xf32> to vector<2xi32>
  spirv.ReturnValue %2 : vector<2xi32>
}
