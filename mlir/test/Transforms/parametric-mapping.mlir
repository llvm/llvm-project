// RUN: mlir-opt -allow-unregistered-dialect -pass-pipeline="builtin.func(test-mapping-to-processing-elements)" %s | FileCheck %s

// CHECK: #[[mul_map:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK: #[[add_map:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>

//      CHECK: func @map1d
// CHECK-SAME: (%[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index)
func @map1d(%lb: index, %ub: index, %step: index) {
  // CHECK: %[[threads:.*]]:2 = "new_processor_id_and_range"() : () -> (index, index)
  %0:2 = "new_processor_id_and_range"() : () -> (index, index)

  // CHECK: %[[thread_offset:.+]] = affine.apply #[[mul_map]]()[%[[threads]]#0, %[[step]]]
  // CHECK: %[[new_lb:.+]] = affine.apply #[[add_map]]()[%[[thread_offset]], %[[lb]]]
  // CHECK: %[[new_step:.+]] = affine.apply #[[mul_map]]()[%[[threads]]#1, %[[step]]]

  // CHECK: scf.for %{{.*}} = %[[new_lb]] to %[[ub]] step %[[new_step]] {
  scf.for %i = %lb to %ub step %step {}
  return
}

//      CHECK: func @map2d
// CHECK-SAME: (%[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index)
func @map2d(%lb : index, %ub : index, %step : index) {
  // CHECK: %[[blocks:.*]]:2 = "new_processor_id_and_range"() : () -> (index, index)
  %0:2 = "new_processor_id_and_range"() : () -> (index, index)

  // CHECK: %[[threads:.*]]:2 = "new_processor_id_and_range"() : () -> (index, index)
  %1:2 = "new_processor_id_and_range"() : () -> (index, index)

  // blockIdx.x * blockDim.x
  // CHECK: %[[bidxXbdimx:.+]] = affine.apply #[[mul_map]]()[%[[blocks]]#0, %[[threads]]#1]
  //
  // threadIdx.x + blockIdx.x * blockDim.x
  // CHECK: %[[tidxpbidxXbdimx:.+]] = affine.apply #[[add_map]]()[%[[bidxXbdimx]], %[[threads]]#0]
  //
  // thread_offset = step * (threadIdx.x + blockIdx.x * blockDim.x)
  // CHECK: %[[thread_offset:.+]] = affine.apply #[[mul_map]]()[%[[tidxpbidxXbdimx]], %[[step]]]
  //
  // new_lb = lb + thread_offset
  // CHECK: %[[new_lb:.+]] = affine.apply #[[add_map]]()[%[[thread_offset]], %[[lb]]]
  //
  // stepXgdimx = step * gridDim.x
  // CHECK: %[[stepXgdimx:.+]] = affine.apply #[[mul_map]]()[%[[blocks]]#1, %[[step]]]
  //
  // new_step = step * gridDim.x * blockDim.x
  // CHECK: %[[new_step:.+]] = affine.apply #[[mul_map]]()[%[[threads]]#1, %[[stepXgdimx]]]
  //
  // CHECK: scf.for %{{.*}} = %[[new_lb]] to %[[ub]] step %[[new_step]] {

  scf.for %i = %lb to %ub step %step {}
  return
}
