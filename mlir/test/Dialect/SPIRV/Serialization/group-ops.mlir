// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK-LABEL: @subgroup_ballot
  func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
    // CHECK: %{{.*}} = spv.SubgroupBallotKHR %{{.*}}: vector<4xi32>
    %0 = spv.SubgroupBallotKHR %predicate: vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
}
