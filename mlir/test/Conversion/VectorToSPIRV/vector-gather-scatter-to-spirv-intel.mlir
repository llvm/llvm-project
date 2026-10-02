// RUN: mlir-opt -split-input-file -test-vector-gather-scatter-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Kernel, Addresses, MaskedGatherScatterINTEL],
               [SPV_INTEL_masked_gather_scatter]>,
    #spirv.resource_limits<>>
} {

// CHECK-LABEL: @vector_gather
//  CHECK-SAME: %[[BASE:.+]]: memref<16xf32, #spirv.storage_class<CrossWorkgroup>>
//  CHECK-SAME: %[[INDICES:.+]]: vector<4xindex>
//  CHECK-SAME: %[[MASK:.+]]: vector<4xi1>
//  CHECK-SAME: %[[PT:.+]]: vector<4xf32>
//   CHECK-DAG: %[[IDXVEC:.+]] = builtin.unrealized_conversion_cast %[[INDICES]] : vector<4xindex> to vector<4xi32>
//       CHECK: %[[EPTR:.+]] = spirv.AccessChain %{{.+}}[%{{.+}}] : !spirv.ptr<!spirv.array<16 x f32>, CrossWorkgroup>, i32
//       CHECK: %[[IDX0:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P0:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX0]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[IDX1:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P1:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX1]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[IDX2:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P2:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX2]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[IDX3:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P3:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX3]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[PTRVEC:.+]] = spirv.CompositeConstruct %[[P0]], %[[P1]], %[[P2]], %[[P3]]
//  CHECK-SAME: -> vector<4x!spirv.ptr<f32, CrossWorkgroup>>
//       CHECK: %[[ALIGN:.+]] = spirv.Constant 0 : i32
//       CHECK: %[[RES:.+]] = spirv.INTEL.MaskedGather %[[PTRVEC]], %[[ALIGN]], %[[MASK]], %[[PT]]
//  CHECK-SAME:   : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32, vector<4xi1>, vector<4xf32> -> vector<4xf32>
//       CHECK: return %[[RES]]
func.func @vector_gather(%base: memref<16xf32, #spirv.storage_class<CrossWorkgroup>>,
                          %indices: vector<4xindex>,
                          %mask: vector<4xi1>,
                          %pass_thru: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %result = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<16xf32, #spirv.storage_class<CrossWorkgroup>>,
      vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: @vector_scatter
//  CHECK-SAME: %[[BASE:.+]]: memref<16xf32, #spirv.storage_class<CrossWorkgroup>>
//  CHECK-SAME: %[[INDICES:.+]]: vector<4xindex>
//  CHECK-SAME: %[[MASK:.+]]: vector<4xi1>
//  CHECK-SAME: %[[VALUES:.+]]: vector<4xf32>
//   CHECK-DAG: %[[IDXVEC:.+]] = builtin.unrealized_conversion_cast %[[INDICES]] : vector<4xindex> to vector<4xi32>
//       CHECK: %[[EPTR:.+]] = spirv.AccessChain %{{.+}}[%{{.+}}] : !spirv.ptr<!spirv.array<16 x f32>, CrossWorkgroup>, i32
//       CHECK: %[[IDX0:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P0:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX0]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[IDX1:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P1:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX1]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[IDX2:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P2:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX2]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[IDX3:.+]] = spirv.VectorExtractDynamic %[[IDXVEC]][%{{.+}}] : vector<4xi32>, i32
//       CHECK: %[[P3:.+]] = spirv.PtrAccessChain %[[EPTR]][%[[IDX3]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK: %[[PTRVEC:.+]] = spirv.CompositeConstruct %[[P0]], %[[P1]], %[[P2]], %[[P3]]
//  CHECK-SAME: -> vector<4x!spirv.ptr<f32, CrossWorkgroup>>
//       CHECK: %[[ALIGN:.+]] = spirv.Constant 0 : i32
//       CHECK: spirv.INTEL.MaskedScatter %[[PTRVEC]], %[[ALIGN]], %[[MASK]], %[[VALUES]]
//  CHECK-SAME:   : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32, vector<4xi1>, vector<4xf32>
//       CHECK: return
func.func @vector_scatter(%base: memref<16xf32, #spirv.storage_class<CrossWorkgroup>>,
                           %indices: vector<4xindex>,
                           %mask: vector<4xi1>,
                           %values: vector<4xf32>) {
  %c0 = arith.constant 0 : index
  vector.scatter %base[%c0][%indices], %mask, %values
    : memref<16xf32, #spirv.storage_class<CrossWorkgroup>>,
      vector<4xindex>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL: @vector_gather_i32
//  CHECK-SAME: %[[BASE:.+]]: memref<16xi32, #spirv.storage_class<CrossWorkgroup>>
//       CHECK: spirv.INTEL.MaskedGather
//  CHECK-SAME: vector<4x!spirv.ptr<i32, CrossWorkgroup>>, i32, vector<4xi1>, vector<4xi32> -> vector<4xi32>
func.func @vector_gather_i32(%base: memref<16xi32, #spirv.storage_class<CrossWorkgroup>>,
                              %indices: vector<4xindex>,
                              %mask: vector<4xi1>,
                              %pass_thru: vector<4xi32>) -> vector<4xi32> {
  %c0 = arith.constant 0 : index
  %result = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<16xi32, #spirv.storage_class<CrossWorkgroup>>,
      vector<4xindex>, vector<4xi1>, vector<4xi32> into vector<4xi32>
  return %result : vector<4xi32>
}

// CHECK-LABEL: @vector_gather_with_alignment
//       CHECK: %[[ALIGN:.+]] = spirv.Constant 4 : i32
//       CHECK: spirv.INTEL.MaskedGather %{{.+}}, %[[ALIGN]]
func.func @vector_gather_with_alignment(%base: memref<16xf32, #spirv.storage_class<CrossWorkgroup>>,
                                         %indices: vector<4xindex>,
                                         %mask: vector<4xi1>,
                                         %pass_thru: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %result = vector.gather %base[%c0][%indices], %mask, %pass_thru
    { alignment = 4 : i64 }
    : memref<16xf32, #spirv.storage_class<CrossWorkgroup>>,
      vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %result : vector<4xf32>
}

} // end module
