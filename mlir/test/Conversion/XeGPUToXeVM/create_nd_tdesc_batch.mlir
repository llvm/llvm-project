// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

// A >2D (batched) create_nd descriptor keeps the innermost matrix as the
// 2D-block surface (base_shape_w/h and pitch come from the innermost two memref
// dims) and encodes the leading (batch) dim element strides into the spare
// payload slots (5..). The matching load/store lowering reads those strides to
// fold the batch offsets into the base pointer, so the batch position stays out
// of the surface. Here memref<4x64x128xf16> has strides [8192, 128, 1]:
//   base_shape_w = 128 (size[2]), base_shape_h = 64 (size[1]),
//   base_pitch   = 128 (stride[1]),  leading stride slot 5 = 8192 (stride[0]).
gpu.module @create_nd_batch {
  // CHECK-LABEL: gpu.func @create_nd_3d
  gpu.func @create_nd_3d(%src: memref<4x64x128xf16>) -> vector<8xi32> {
    // CHECK: %[[W:.+]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[H:.+]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[PITCH:.+]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[P0:.+]] = vector.insert %{{.*}}, %{{.*}} [0] : i64 into vector<4xi64>
    // CHECK: %[[P1:.+]] = vector.bitcast %[[P0]] : vector<4xi64> to vector<8xi32>
    // CHECK: %[[P2:.+]] = vector.insert %[[W]], %[[P1]] [2] : i32 into vector<8xi32>
    // CHECK: %[[P3:.+]] = vector.insert %[[H]], %[[P2]] [3] : i32 into vector<8xi32>
    // CHECK: %[[P4:.+]] = vector.insert %[[PITCH]], %[[P3]] [4] : i32 into vector<8xi32>
    // CHECK: %[[C8192:.+]] = arith.constant 8192 : i64
    // CHECK: %[[LS0:.+]] = arith.trunci %[[C8192]] : i64 to i32
    // CHECK: %{{.+}} = vector.insert %[[LS0]], %[[P4]] [5] : i32 into vector<8xi32>
    %t = xegpu.create_nd_tdesc %src : memref<4x64x128xf16> -> !xegpu.tensor_desc<1x8x16xf16>
    %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<1x8x16xf16> to vector<8xi32>
    gpu.return %c : vector<8xi32>
  }
}
