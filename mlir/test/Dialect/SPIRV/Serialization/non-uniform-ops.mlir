// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK-LABEL: @group_non_uniform_ballot
  func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> {
    // CHECK: %{{.*}} = spv.GroupNonUniformBallot "Workgroup" %{{.*}}: vector<4xi32>
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
}
