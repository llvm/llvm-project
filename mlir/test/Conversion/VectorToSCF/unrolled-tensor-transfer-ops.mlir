// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf{full-unroll=true lower-tensors=true}))" -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @transfer_read_2d(
//       CHECK: %[[V_INIT:.*]] = arith.constant dense<-4.200000e+01> : vector<4x9xf32>
//       CHECK: %[[V0:.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} {in_bounds = [true]} : tensor<?x?xf32>, vector<9xf32>
//       CHECK: %[[I0:.*]] = vector.insert %[[V0]], %[[V_INIT]] [0] : vector<9xf32> into vector<4x9xf32>
//       CHECK: %[[V1:.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} {in_bounds = [true]} : tensor<?x?xf32>, vector<9xf32>
//       CHECK: %[[I1:.*]] = vector.insert %[[V1]], %[[I0]] [1] : vector<9xf32> into vector<4x9xf32>
//       CHECK: %[[V2:.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} {in_bounds = [true]} : tensor<?x?xf32>, vector<9xf32>
//       CHECK: %[[I2:.*]] = vector.insert %[[V2]], %[[I1]] [2] : vector<9xf32> into vector<4x9xf32>
//       CHECK: %[[V3:.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} {in_bounds = [true]} : tensor<?x?xf32>, vector<9xf32>
//       CHECK: %[[I3:.*]] = vector.insert %[[V3]], %[[I2]] [3] : vector<9xf32> into vector<4x9xf32>
//       CHECK: return %[[I3]] : vector<4x9xf32>
func.func @transfer_read_2d(%A : tensor<?x?xf32>, %base1 : index, %base2 : index)
    -> (vector<4x9xf32>){
  %p = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %p {in_bounds = [true, true]}
      : tensor<?x?xf32>, vector<4x9xf32>
  return %f : vector<4x9xf32>
}

// -----

// CHECK-LABEL: func @transfer_write_2d(
//       CHECK:   %[[V0:.*]] = vector.extract %{{.*}}[0] : vector<3xf32> from vector<2x3xf32>
//       CHECK:   %[[T0:.*]] = vector.transfer_write %[[V0]], %{{.*}}[{{.*}}] {in_bounds = [true]} : vector<3xf32>, tensor<?x?xf32>
//       CHECK:   %[[V1:.*]] = vector.extract %{{.*}}[1] : vector<3xf32> from vector<2x3xf32>
//       CHECK:   %[[T1:.*]] = vector.transfer_write %[[V1]], %[[T0]][{{.*}}] {in_bounds = [true]} : vector<3xf32>, tensor<?x?xf32>
//       CHECK:   return %[[T1]] : tensor<?x?xf32>
func.func @transfer_write_2d(%A : tensor<?x?xf32>, %vec : vector<2x3xf32>,
                        %base1 : index, %base2 : index) -> (tensor<?x?xf32>) {
  %t = vector.transfer_write %vec, %A[%base1, %base2] {in_bounds = [true, true]}
      : vector<2x3xf32>, tensor<?x?xf32>
  return %t : tensor<?x?xf32>
}

