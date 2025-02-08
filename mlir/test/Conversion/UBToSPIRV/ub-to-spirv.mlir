// RUN: mlir-opt -split-input-file -convert-ub-to-spirv -verify-diagnostics %s | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64, Shader], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @check_poison
func.func @check_poison() {
// CHECK: {{.*}} = spirv.Undef : i32
  %0 = ub.poison : index
// CHECK: {{.*}} = spirv.Undef : i16
  %1 = ub.poison : i16
// CHECK: {{.*}} = spirv.Undef : f64
  %2 = ub.poison : f64
// TODO: vector is not covered yet
// CHECK: {{.*}} = ub.poison : vector<4xf32>
  %3 = ub.poison : vector<4xf32>
  return
}

}
