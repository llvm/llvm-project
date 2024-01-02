// RUN: mlir-opt --allow-unregistered-dialect --test-gpu-subgroup-reduce-lowering %s | FileCheck %s

// CHECK: gpu.module @kernels {
gpu.module @kernels {

  // CHECK-LABEL: gpu.func @kernel0(
  // CHECK-SAME: %[[ARG0:.+]]: vector<5xf16>)
  gpu.func @kernel0(%arg0: vector<5xf16>) kernel {
    // CHECK: %[[VZ:.+]] = arith.constant dense<0.0{{.*}}> : vector<5xf16>
    // CHECK: %[[E0:.+]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0], sizes = [2], strides = [1]} : vector<5xf16> to vector<2xf16>
    // CHECK: %[[R0:.+]] = gpu.subgroup_reduce add %[[E0]] : (vector<2xf16>) -> vector<2xf16>
    // CHECK: %[[V0:.+]] = vector.insert_strided_slice %[[R0]], %[[VZ]] {offsets = [0], strides = [1]} : vector<2xf16> into vector<5xf16>
    // CHECK: %[[E1:.+]] = vector.extract_strided_slice %[[ARG0]] {offsets = [2], sizes = [2], strides = [1]} : vector<5xf16> to vector<2xf16>
    // CHECK: %[[R1:.+]] = gpu.subgroup_reduce add %[[E1]] : (vector<2xf16>) -> vector<2xf16>
    // CHECK: %[[V1:.+]] = vector.insert_strided_slice %[[R1]], %[[V0]] {offsets = [2], strides = [1]} : vector<2xf16> into vector<5xf16>
    // CHECK: %[[E2:.+]] = vector.extract %[[ARG0]][4] : f16 from vector<5xf16>
    // CHECK: %[[R2:.+]] = gpu.subgroup_reduce add %[[E2]] : (f16) -> f16
    // CHECK: %[[V2:.+]] = vector.insert %[[R2]], %[[V1]] [4] : f16 into vector<5xf16>
    // CHECK: "test.consume"(%[[V2]]) : (vector<5xf16>) -> ()
    %sum0 = gpu.subgroup_reduce add %arg0 : (vector<5xf16>) -> (vector<5xf16>)
    "test.consume"(%sum0) : (vector<5xf16>) -> ()


    // CHECK-COUNT-3: gpu.subgroup_reduce mul {{.+}} uniform
    // CHECK: "test.consume"
    %sum1 = gpu.subgroup_reduce mul %arg0 uniform : (vector<5xf16>) -> (vector<5xf16>)
    "test.consume"(%sum1) : (vector<5xf16>) -> ()

    // CHECK: gpu.return
    gpu.return
  }

  // CHECK-LABEL: gpu.func @kernel1(
  // CHECK-SAME: %[[ARG0:.+]]: vector<1xf32>)
  gpu.func @kernel1(%arg0: vector<1xf32>) kernel {
    // CHECK: %[[E0:.+]] = vector.extract %[[ARG0]][0] : f32 from vector<1xf32>
    // CHECK: %[[R0:.+]] = gpu.subgroup_reduce add %[[E0]] : (f32) -> f32
    // CHECK: %[[V0:.+]] = vector.broadcast %[[R0]] : f32 to vector<1xf32>
    // CHECK: "test.consume"(%[[V0]]) : (vector<1xf32>) -> ()
    %sum0 = gpu.subgroup_reduce add %arg0 : (vector<1xf32>) -> (vector<1xf32>)
    "test.consume"(%sum0) : (vector<1xf32>) -> ()

    // CHECK: gpu.subgroup_reduce add {{.+}} uniform : (f32) -> f32
    // CHECK: "test.consume"
    %sum1 = gpu.subgroup_reduce add %arg0 uniform : (vector<1xf32>) -> (vector<1xf32>)
    "test.consume"(%sum1) : (vector<1xf32>) -> ()

    // CHECK: gpu.return
    gpu.return
  }

  // These vectors fit the native shuffle size and should not be broken down.
  //
  // CHECK-LABEL: gpu.func @kernel2(
  // CHECK-SAME: %[[ARG0:.+]]: vector<3xi8>, %[[ARG1:.+]]: vector<4xi8>)
  gpu.func @kernel2(%arg0: vector<3xi8>, %arg1: vector<4xi8>) kernel {
    // CHECK: %[[R0:.+]] = gpu.subgroup_reduce add %[[ARG0]] : (vector<3xi8>) -> vector<3xi8>
    // CHECK: "test.consume"(%[[R0]]) : (vector<3xi8>) -> ()
    %sum0 = gpu.subgroup_reduce add %arg0 : (vector<3xi8>) -> (vector<3xi8>)
    "test.consume"(%sum0) : (vector<3xi8>) -> ()

    // CHECK: %[[R1:.+]] = gpu.subgroup_reduce add %[[ARG1]] : (vector<4xi8>) -> vector<4xi8>
    // CHECK: "test.consume"(%[[R1]]) : (vector<4xi8>) -> ()
    %sum1 = gpu.subgroup_reduce add %arg1 : (vector<4xi8>) -> (vector<4xi8>)
    "test.consume"(%sum1) : (vector<4xi8>) -> ()

    // CHECK: gpu.return
    gpu.return
  }

}
