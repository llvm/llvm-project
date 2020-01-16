// RUN: mlir-opt -disable-pass-threading -test-spirv-target-env %s | FileCheck %s

// Note: The following tests check that a spv.target_env can properly control
// the conversion target and filter unavailable ops during the conversion.
// We don't care about the op argument consistency too much; so certain enum
// values for enum attributes may not make much sense for the test op.

// spv.AtomicCompareExchangeWeak is available from SPIR-V 1.0 to 1.3 under
// Kernel capability.
// spv.AtomicCompareExchangeWeak has two memory semantics enum attribute,
// whose value, if containing AtomicCounterMemory bit, additionally requires
// AtomicStorage capability.

// spv.GroupNonUniformBallot is available starting from SPIR-V 1.3 under
// GroupNonUniform capability.

// spv.SubgroupBallotKHR is available under in all SPIR-V versions under
// SubgroupBallotKHR capability and SPV_KHR_shader_ballot extension.

// Enum case symbol (value) map:
// Version: 1.0 (0), 1.1 (1), 1.2 (2), 1.3 (3), 1.4 (4)
// Capability: Kernel (6), AtomicStorage (21), GroupNonUniformBallot (64),
//             SubgroupBallotKHR (4423)

//===----------------------------------------------------------------------===//
// MaxVersion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmp_exchange_weak_suitable_version_capabilities
func @cmp_exchange_weak_suitable_version_capabilities(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = {version = 1: i32, extensions = [], capabilities = [6: i32, 21: i32]}
} {
  // CHECK: spv.AtomicCompareExchangeWeak "Workgroup" "AcquireRelease|AtomicCounterMemory" "Acquire"
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @cmp_exchange_weak_unsupported_version
func @cmp_exchange_weak_unsupported_version(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = {version = 4: i32, extensions = [], capabilities = [6: i32, 21: i32]}
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

//===----------------------------------------------------------------------===//
// MinVersion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_ballot_suitable_version
func @group_non_uniform_ballot_suitable_version(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = {version = 4: i32, extensions = [], capabilities = [64: i32]}
} {
  // CHECK: spv.GroupNonUniformBallot "Workgroup"
  %0 = "test.convert_to_group_non_uniform_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @group_non_uniform_ballot_unsupported_version
func @group_non_uniform_ballot_unsupported_version(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = {version = 1: i32, extensions = [], capabilities = [64: i32]}
} {
  // CHECK: test.convert_to_group_non_uniform_ballot_op
  %0 = "test.convert_to_group_non_uniform_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

//===----------------------------------------------------------------------===//
// Capability
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmp_exchange_weak_missing_capability_kernel
func @cmp_exchange_weak_missing_capability_kernel(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = {version = 3: i32, extensions = [], capabilities = [21: i32]}
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @cmp_exchange_weak_missing_capability_atomic_storage
func @cmp_exchange_weak_missing_capability_atomic_storage(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = {version = 3: i32, extensions = [], capabilities = [6: i32]}
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @subgroup_ballot_missing_capability
func @subgroup_ballot_missing_capability(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = {version = 4: i32, extensions = ["SPV_KHR_shader_ballot"], capabilities = []}
} {
  // CHECK: test.convert_to_subgroup_ballot_op
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @subgroup_ballot_suitable_extension
func @subgroup_ballot_suitable_extension(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = {version = 4: i32, extensions = ["SPV_KHR_shader_ballot"], capabilities = [4423: i32]}
} {
  // CHECK: spv.SubgroupBallotKHR
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @subgroup_ballot_missing_extension
func @subgroup_ballot_missing_extension(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = {version = 4: i32, extensions = [], capabilities = [4423: i32]}
} {
  // CHECK: test.convert_to_subgroup_ballot_op
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}
