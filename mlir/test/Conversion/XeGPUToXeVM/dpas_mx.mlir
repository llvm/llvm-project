// RUN: mlir-opt --split-input-file -convert-xegpu-to-xevm %s | FileCheck %s

// CHECK-LABEL: gpu.func @dpas_mx_bf8
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xf8E5M2>, %[[ARG1:.*]]: vector<32xf8E5M2>,
// CHECK-SAME: %[[ARG2:.*]]: vector<8xf32>,
// CHECK-SAME: %[[ARG3:.*]]: vector<1xf8E8M0FNU>, %[[ARG4:.*]]: vector<1xf8E8M0FNU>)
gpu.module @dpas_mx_bf8 [#xevm.target<chip = "cri">] {
  gpu.func @dpas_mx_bf8(%a: vector<16xf8E5M2>, %b: vector<32xf8E5M2>, %acc: vector<8xf32>,
                         %scale_a: vector<1xf8E8M0FNU>, %scale_b: vector<1xf8E8M0FNU>) kernel {
    // CHECK: %[[V0:.*]] = vector.extract %[[ARG4]][0] : f8E8M0FNU from vector<1xf8E8M0FNU>
    // CHECK: %[[V1:.*]] = arith.bitcast %[[V0]] : f8E8M0FNU to i8
    // CHECK: %[[V2:.*]] = vector.extract %[[ARG3]][0] : f8E8M0FNU from vector<1xf8E8M0FNU>
    // CHECK: %[[V3:.*]] = arith.bitcast %[[V2]] : f8E8M0FNU to i8
    // CHECK: %[[V4:.*]] = vector.bitcast %[[ARG1]] : vector<32xf8E5M2> to vector<32xi8>
    // CHECK: %[[V5:.*]] = vector.bitcast %[[ARG0]] : vector<16xf8E5M2> to vector<16xi8>
    // CHECK: %{{.*}} = xevm.mma_mx %[[V5]], %[[V4]], %[[V3]], %[[V1]], %[[ARG2]]
    // CHECK-SAME: {shape = <m = 8, n = 16, k = 32>, types = <d = f32, a = bf8, b = bf8, c = f32>}
    // CHECK-SAME: : (vector<16xi8>, vector<32xi8>, i8, i8, vector<8xf32>) -> vector<8xf32>
    %res = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a scale_b = %scale_b :
        vector<16xf8E5M2>, vector<32xf8E5M2>, vector<8xf32>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> -> vector<8xf32>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @dpas_mx_f8
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xf8E4M3FN>, %[[ARG1:.*]]: vector<32xf8E4M3FN>,
// CHECK-SAME: %[[ARG2:.*]]: vector<8xf32>,
// CHECK-SAME: %[[ARG3:.*]]: vector<1xf8E8M0FNU>, %[[ARG4:.*]]: vector<1xf8E8M0FNU>)
gpu.module @dpas_mx_f8 [#xevm.target<chip = "cri">] {
  gpu.func @dpas_mx_f8(%a: vector<16xf8E4M3FN>, %b: vector<32xf8E4M3FN>, %acc: vector<8xf32>,
                         %scale_a: vector<1xf8E8M0FNU>, %scale_b: vector<1xf8E8M0FNU>) kernel {
    // CHECK: %[[V0:.*]] = vector.extract %[[ARG4]][0] : f8E8M0FNU from vector<1xf8E8M0FNU>
    // CHECK: %[[V1:.*]] = arith.bitcast %[[V0]] : f8E8M0FNU to i8
    // CHECK: %[[V2:.*]] = vector.extract %[[ARG3]][0] : f8E8M0FNU from vector<1xf8E8M0FNU>
    // CHECK: %[[V3:.*]] = arith.bitcast %[[V2]] : f8E8M0FNU to i8
    // CHECK: %[[V4:.*]] = vector.bitcast %[[ARG1]] : vector<32xf8E4M3FN> to vector<32xi8>
    // CHECK: %[[V5:.*]] = vector.bitcast %[[ARG0]] : vector<16xf8E4M3FN> to vector<16xi8>
    // CHECK: %{{.*}} = xevm.mma_mx %[[V5]], %[[V4]], %[[V3]], %[[V1]], %[[ARG2]]
    // CHECK-SAME: {shape = <m = 8, n = 16, k = 32>, types = <d = f32, a = f8, b = f8, c = f32>}
    // CHECK-SAME: : (vector<16xi8>, vector<32xi8>, i8, i8, vector<8xf32>) -> vector<8xf32>
    %res = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a scale_b = %scale_b :
        vector<16xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf32>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> -> vector<8xf32>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @dpas_mx_e2m1
// CHECK-SAME: (%[[ARG0:.*]]: vector<32xf4E2M1FN>, %[[ARG1:.*]]: vector<64xf4E2M1FN>,
// CHECK-SAME: %[[ARG2:.*]]: vector<8xf32>,
// CHECK-SAME: %[[ARG3:.*]]: vector<2xf8E8M0FNU>, %[[ARG4:.*]]: vector<2xf8E8M0FNU>)
gpu.module @dpas_mx_e2m1 [#xevm.target<chip = "cri">] {
  gpu.func @dpas_mx_e2m1(%a: vector<32xf4E2M1FN>, %b: vector<64xf4E2M1FN>, %acc: vector<8xf32>,
                         %scale_a: vector<2xf8E8M0FNU>, %scale_b: vector<2xf8E8M0FNU>) kernel {
    // CHECK: %[[V0:.*]] = vector.bitcast %[[ARG4]] : vector<2xf8E8M0FNU> to vector<2xi8>
    // CHECK: %[[V1:.*]] = vector.bitcast %[[ARG3]] : vector<2xf8E8M0FNU> to vector<2xi8>
    // CHECK: %[[V2:.*]] = vector.bitcast %[[ARG1]] : vector<64xf4E2M1FN> to vector<64xi4>
    // CHECK: %[[V3:.*]] = vector.bitcast %[[ARG0]] : vector<32xf4E2M1FN> to vector<32xi4>
    // CHECK: %[[V4:.*]] = vector.bitcast %[[V3]] : vector<32xi4> to vector<16xi8>
    // CHECK: %[[V5:.*]] = vector.bitcast %[[V2]] : vector<64xi4> to vector<32xi8>
    // CHECK: %{{.*}} = xevm.mma_mx %[[V4]], %[[V5]], %[[V1]], %[[V0]], %[[ARG2]]
    // CHECK-SAME: {shape = <m = 8, n = 16, k = 64>, types = <d = f32, a = e2m1, b = e2m1, c = f32>}
    // CHECK-SAME: : (vector<16xi8>, vector<32xi8>, vector<2xi8>, vector<2xi8>, vector<8xf32>) -> vector<8xf32>
    %res = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a scale_b = %scale_b :
        vector<32xf4E2M1FN>, vector<64xf4E2M1FN>, vector<8xf32>, vector<2xf8E8M0FNU>, vector<2xf8E8M0FNU> -> vector<8xf32>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @dpas_mx_no_acc
gpu.module @dpas_mx_no_acc [#xevm.target<chip = "cri">] {
  gpu.func @dpas_mx_no_acc(%a: vector<32xf4E2M1FN>, %b: vector<64xf4E2M1FN>,
                         %scale_a: vector<2xf8E8M0FNU>, %scale_b: vector<2xf8E8M0FNU>) kernel {
    // CHECK: %[[ACC:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
    // CHECK: %{{.*}} = xevm.mma_mx %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ACC]]
    %res = xegpu.dpas_mx %a, %b scale_a = %scale_a scale_b = %scale_b :
        vector<32xf4E2M1FN>, vector<64xf4E2M1FN>, vector<8xf32>, vector<2xf8E8M0FNU>, vector<2xf8E8M0FNU> -> vector<8xf32>
    gpu.return
  }
}
