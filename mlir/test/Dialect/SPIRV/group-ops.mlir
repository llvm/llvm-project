// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.SubgroupBallotKHR
//===----------------------------------------------------------------------===//

func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: %{{.*}} = spv.SubgroupBallotKHR %{{.*}} : vector<4xi32>
  %0 = spv.SubgroupBallotKHR %predicate: vector<4xi32>
  return %0: vector<4xi32>
}
