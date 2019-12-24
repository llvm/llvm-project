// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformBallot
//===----------------------------------------------------------------------===//

func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: %{{.*}} = spv.GroupNonUniformBallot "Workgroup" %{{.*}}: vector<4xi32>
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// -----

func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spv.GroupNonUniformBallot "Device" %predicate : vector<4xi32>
  return %0: vector<4xi32>
}
