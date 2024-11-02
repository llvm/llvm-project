// RUN: mlir-opt -split-input-file -convert-linalg-to-spirv -canonicalize -verify-diagnostics %s -o - | FileCheck %s

//===----------------------------------------------------------------------===//
// Single workgroup reduction
//===----------------------------------------------------------------------===//

#single_workgroup_reduction_trait = {
  iterator_types = ["reduction"],
  indexing_maps = [
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (0)>
  ]
}

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniformArithmetic], []>, #spirv.resource_limits<>>
} {

// CHECK:      spirv.GlobalVariable
// CHECK-SAME: built_in("LocalInvocationId")

// CHECK:      @single_workgroup_reduction
// CHECK-SAME: (%[[INPUT:.+]]: !spirv.ptr{{.+}}, %[[OUTPUT:.+]]: !spirv.ptr{{.+}})

// CHECK:        %[[ZERO:.+]] = spirv.Constant 0 : i32
// CHECK:        %[[ID:.+]] = spirv.Load "Input" %{{.+}} : vector<3xi32>
// CHECK:        %[[X:.+]] = spirv.CompositeExtract %[[ID]][0 : i32]

// CHECK:        %[[INPTR:.+]] = spirv.AccessChain %[[INPUT]][%[[ZERO]], %[[X]]]
// CHECK:        %[[VAL:.+]] = spirv.Load "StorageBuffer" %[[INPTR]] : i32
// CHECK:        %[[ADD:.+]] = spirv.GroupNonUniformIAdd "Subgroup" "Reduce" %[[VAL]] : i32

// CHECK:        %[[OUTPTR:.+]] = spirv.AccessChain %[[OUTPUT]][%[[ZERO]], %[[ZERO]]]
// CHECK:        %[[ELECT:.+]] = spirv.GroupNonUniformElect <Subgroup> : i1

// CHECK:        spirv.mlir.selection {
// CHECK:          spirv.BranchConditional %[[ELECT]], ^bb1, ^bb2
// CHECK:        ^bb1:
// CHECK:          spirv.AtomicIAdd "Device" "AcquireRelease" %[[OUTPTR]], %[[ADD]]
// CHECK:          spirv.Branch ^bb2
// CHECK:        ^bb2:
// CHECK:          spirv.mlir.merge
// CHECK:        }
// CHECK:        spirv.Return

func.func @single_workgroup_reduction(%input: memref<16xi32, #spirv.storage_class<StorageBuffer>>, %output: memref<1xi32, #spirv.storage_class<StorageBuffer>>) attributes {
  spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>
} {
  linalg.generic #single_workgroup_reduction_trait
      ins(%input : memref<16xi32, #spirv.storage_class<StorageBuffer>>)
     outs(%output : memref<1xi32, #spirv.storage_class<StorageBuffer>>) {
    ^bb(%in: i32, %out: i32):
      %sum = arith.addi %in, %out : i32
      linalg.yield %sum : i32
  }
  spirv.Return
}
}

// -----

// Missing shader entry point ABI

#single_workgroup_reduction_trait = {
  iterator_types = ["reduction"],
  indexing_maps = [
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (0)>
  ]
}

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniformArithmetic], []>, #spirv.resource_limits<>>
} {
func.func @single_workgroup_reduction(%input: memref<16xi32, #spirv.storage_class<StorageBuffer>>, %output: memref<1xi32, #spirv.storage_class<StorageBuffer>>) {
  // expected-error @+1 {{failed to legalize operation 'linalg.generic'}}
  linalg.generic #single_workgroup_reduction_trait
      ins(%input : memref<16xi32, #spirv.storage_class<StorageBuffer>>)
     outs(%output : memref<1xi32, #spirv.storage_class<StorageBuffer>>) {
    ^bb(%in: i32, %out: i32):
      %sum = arith.addi %in, %out : i32
      linalg.yield %sum : i32
  }
  return
}
}

// -----

// Mismatch between shader entry point ABI and input memref shape

#single_workgroup_reduction_trait = {
  iterator_types = ["reduction"],
  indexing_maps = [
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (0)>
  ]
}

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniformArithmetic], []>, #spirv.resource_limits<>>
} {
func.func @single_workgroup_reduction(%input: memref<16xi32, #spirv.storage_class<StorageBuffer>>, %output: memref<1xi32, #spirv.storage_class<StorageBuffer>>) attributes {
  spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 1, 1]>: vector<3xi32>>
} {
  // expected-error @+1 {{failed to legalize operation 'linalg.generic'}}
  linalg.generic #single_workgroup_reduction_trait
      ins(%input : memref<16xi32, #spirv.storage_class<StorageBuffer>>)
     outs(%output : memref<1xi32, #spirv.storage_class<StorageBuffer>>) {
    ^bb(%in: i32, %out: i32):
      %sum = arith.addi %in, %out : i32
      linalg.yield %sum : i32
  }
  spirv.Return
}
}

// -----

// Unsupported multi-dimension input memref

#single_workgroup_reduction_trait = {
  iterator_types = ["parallel", "reduction"],
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i)>
  ]
}

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniformArithmetic], []>, #spirv.resource_limits<>>
} {
func.func @single_workgroup_reduction(%input: memref<16x8xi32, #spirv.storage_class<StorageBuffer>>, %output: memref<16xi32, #spirv.storage_class<StorageBuffer>>) attributes {
  spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 8, 1]>: vector<3xi32>>
} {
  // expected-error @+1 {{failed to legalize operation 'linalg.generic'}}
  linalg.generic #single_workgroup_reduction_trait
      ins(%input : memref<16x8xi32, #spirv.storage_class<StorageBuffer>>)
     outs(%output : memref<16xi32, #spirv.storage_class<StorageBuffer>>) {
    ^bb(%in: i32, %out: i32):
      %sum = arith.addi %in, %out : i32
      linalg.yield %sum : i32
  }
  spirv.Return
}
}
