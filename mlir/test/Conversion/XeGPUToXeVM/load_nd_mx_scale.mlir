// RUN: mlir-opt --split-input-file -convert-xegpu-to-xevm %s | FileCheck %s

// CHECK-LABEL: gpu.func @load_nd_mx_scale
gpu.module @load_nd_mx_scale [#xevm.target<chip = "cri">] {
  gpu.func @load_nd_mx_scale(%scale_a: memref<128x32xf8E8M0FNU>, %scale_b: memref<32x256xf8E8M0FNU>) kernel {
    %scale_a_desc_1 = xegpu.create_nd_tdesc %scale_a : memref<128x32xf8E8M0FNU> -> !xegpu.tensor_desc<8x2xf8E8M0FNU>
    %scale_a_desc_2 = xegpu.create_nd_tdesc %scale_a : memref<128x32xf8E8M0FNU> -> !xegpu.tensor_desc<8x1xf8E8M0FNU>
    // CHECK: %[[BASE_1:.*]] = vector.extract {{.*}}[0] : i64 from vector<4xi64>
    // CHECK: %[[PITCH_1:.*]] = llvm.sext {{.*}} : i32 to i64
    // CHECK: %[[OFF_W:.*]] = arith.constant 5 : i64
    // CHECK: %[[OFF_H:.*]] = arith.constant 3 : i64
    // CHECK: %[[ID_X_1:.*]] = xevm.lane_id : i64
    // CHECK: %[[EIGHT:.*]] = llvm.mlir.constant(8 : i64) : i64
    // CHECK: %[[ID_1:.*]] = llvm.srem %[[ID_X_1]], %[[EIGHT]] : i64
    // CHECK: %[[Y1:.*]] = llvm.add %[[OFF_H]], %[[ID_1]] : i64
    // CHECK: %[[TMP1:.*]] = llvm.mul %[[PITCH_1]], %[[Y1]] : i64
    // CHECK: %[[TMP2:.*]] = llvm.add %[[TMP1]], %[[OFF_W]] : i64
    // CHECK: %[[TMP3:.*]] = llvm.add %[[BASE_1]], %[[TMP2]] : i64
    // CHECK: %[[PTR1:.*]] = llvm.inttoptr %[[TMP3]] : i64 to !llvm.ptr<1>
    // CHECK: %[[SCALE_A_1:.*]] = llvm.load %[[PTR1]] : !llvm.ptr<1> -> vector<2xi8>
    %scale_a_1 = xegpu.load_nd %scale_a_desc_1[3, 5] : !xegpu.tensor_desc<8x2xf8E8M0FNU> -> vector<2xf8E8M0FNU>
    // CHECK: %[[OFF_W_2:.*]] = arith.constant 5 : i64
    // CHECK: %[[TMP4:.*]] = llvm.add {{.*}}, %[[OFF_W_2]] : i64
    // CHECK: %[[TMP5:.*]] = llvm.add {{.*}}, %[[TMP4]] : i64
    // CHECK: %[[PTR2:.*]] = llvm.inttoptr %[[TMP5]] : i64 to !llvm.ptr<1>
    // CHECK: %[[SCALE_A_2:.*]] = llvm.load %[[PTR2]] : !llvm.ptr<1> -> i8
    %scale_a_2 = xegpu.load_nd %scale_a_desc_2[3, 5] : !xegpu.tensor_desc<8x1xf8E8M0FNU> -> vector<1xf8E8M0FNU>
    %scale_b_desc_1 = xegpu.create_nd_tdesc %scale_b : memref<32x256xf8E8M0FNU> -> !xegpu.tensor_desc<2x16xf8E8M0FNU>
    %scale_b_desc_2 = xegpu.create_nd_tdesc %scale_b : memref<32x256xf8E8M0FNU> -> !xegpu.tensor_desc<1x16xf8E8M0FNU>
    // CHECK: %[[PITCH_3:.*]] = llvm.sext {{.*}} : i32 to i64
    // CHECK: %[[OFF_W_3:.*]] = arith.constant 5 : i64
    // CHECK: %[[OFF_H_3:.*]] = arith.constant 3 : i64
    // CHECK: %[[ID_X_3:.*]] = xevm.lane_id : i64
    // CHECK: %[[TMP6:.*]] = llvm.add %[[OFF_W_3]], %[[ID_X_3]] : i64
    // CHECK: %[[TMP7:.*]] = llvm.mul {{.*}}, %[[OFF_H_3]] : i64
    // CHECK: %[[TMP8:.*]] = llvm.add %[[TMP7]], %[[TMP6]] : i64
    // CHECK: %[[TMP9:.*]] = llvm.add {{.*}}, %[[TMP8:.*]] : i64
    // CHECK: %[[PTR3:.*]] = llvm.inttoptr %[[TMP9]] : i64 to !llvm.ptr<1>
    // CHECK: %[[FIRST:.*]] = llvm.load %[[PTR3]] : !llvm.ptr<1> -> i8
    // CHECK: %[[TMP_RES:.*]] = vector.insert %[[FIRST]], {{.*}} [0] : i8 into vector<2xi8>
    // CHECK: %[[TMP10:.*]] = llvm.add %[[TMP9]], %[[PITCH_3]] : i64
    // CHECK: %[[PTR3_1:.*]] = llvm.inttoptr %[[TMP10]] : i64 to !llvm.ptr<1>
    // CHECK: %[[SECOND:.*]] = llvm.load %[[PTR3_1]] : !llvm.ptr<1> -> i8
    // CHECK: %[[SCALE_B_1:.*]] = vector.insert %[[SECOND]], %[[TMP_RES]] [1] : i8 into vector<2xi8>
    %scale_b_1 = xegpu.load_nd %scale_b_desc_1[3, 5] : !xegpu.tensor_desc<2x16xf8E8M0FNU> -> vector<2xf8E8M0FNU>
    // CHECK: %[[PTR4:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<1>
    // CHECK: %[[SCALE_B_2:.*]]7 = llvm.load %[[PTR4]] : !llvm.ptr<1> -> i8
    %scale_b_2 = xegpu.load_nd %scale_b_desc_2[3, 5] : !xegpu.tensor_desc<1x16xf8E8M0FNU> -> vector<1xf8E8M0FNU>
    gpu.return
  }
}
