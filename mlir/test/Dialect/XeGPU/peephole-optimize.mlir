// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=pvc'  \
// RUN:   --xegpu-optimize-peephole --canonicalize --cse --split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @no_scf(
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]: memref<64x64xf16>, %{{.*}}: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK:         %[[C16:.*]] = arith.constant 16 : index
// CHECK:         %[[C32:.*]] = arith.constant 32 : index
// CHECK:         %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<64x64xf16> -> index
// CHECK:         %[[T0:.*]] = arith.index_cast %[[PTR]] : index to i64
// CHECK:         %[[BDESC:.*]] = xegpu.create_nd_tdesc %[[T0]], shape : [64, %[[C32]]], strides : [%[[C32]], 1] : i64
// CHECK-SAME:      -> !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
// CHECK-NEXT:    %[[B:.*]] = xegpu.load_nd %[[BDESC]][%{{.*}}, %[[C16]]]
// CHECK-SAME:      {layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
// CHECK-SAME:      : !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<16x8xi32>
// CHECK:         %[[BITCAST:.*]] = vector.bitcast %[[B]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>} : vector<16x8xi32> to vector<16x16xf16>
#a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#b = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
#bt = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>
gpu.module @xevm_module {
gpu.func @no_scf(%arg0: memref<64x64xf16>, %arg1: vector<8x16xf16>) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #b>
  %1 = xegpu.load_nd %0[%c0, %c32] { result_layout = #b } : !xegpu.tensor_desc<16x16xf16, #b> -> vector<16x16xf16>
  %2 = vector.transpose %1, [1, 0] { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
  %6 = xegpu.dpas %arg1, %2 { layout_result_0 = #a } : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  gpu.return %6 : vector<8x16xf32>
}
}

// -----
// CHECK-LABEL: gpu.func @no_scf_i8(
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]: memref<64x64xi8>, %{{.*}}: vector<8x32xi8>) -> vector<8x16xi32> {
// CHECK:         %[[C16:.*]] = arith.constant 16 : index
// CHECK:         %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<64x64xi8> -> index
// CHECK:         %[[T0:.*]] = arith.index_cast %[[PTR]] : index to i64
// CHECK:         %[[T1:.*]] = xegpu.create_nd_tdesc %[[T0]], shape : [64, %[[C16]]], strides : [%[[C16]], 1] : i64
// CHECK-SAME:      -> !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
// CHECK:         %[[T2:.*]] = xegpu.load_nd %[[T1]][%{{.*}}, %[[C16]]]
// CHECK-SAME:      {layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
// CHECK-SAME:      : !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<16x8xi32>
// CHECK:         %[[T3:.*]] = vector.bitcast %[[T2]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 4]>} : vector<16x8xi32> to vector<16x32xi8>
#a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>
#b = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 4]>
#bt = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>
#c = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
gpu.module @xevm_module {
gpu.func @no_scf_i8(%arg0: memref<64x64xi8>, %arg1: vector<8x32xi8>) -> vector<8x16xi32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<64x64xi8> -> !xegpu.tensor_desc<16x32xi8, #b>
  %1 = xegpu.load_nd %0[%c0, %c64] { result_layout = #b } : !xegpu.tensor_desc<16x32xi8, #b> -> vector<16x32xi8>
  %2 = vector.transpose %1, [1, 0] { layout_result_0 = #bt } : vector<16x32xi8> to vector<32x16xi8>
  %6 = xegpu.dpas %arg1, %2 { layout_result_0 = #c } : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
  gpu.return %6 : vector<8x16xi32>
}
}


// -----
// CHECK-LABEL:   gpu.func @gemm_b_transpose(
// CHECK-SAME:      %{{.*}} memref<256x256xf16>, %[[ARG1:[a-zA-Z0-9]+]]: memref<256x256xf16>, %{{.*}}: memref<256x256xf32>) {
// CHECK:           %[[C128:.*]] = arith.constant 128 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[C256:.*]] = arith.constant 256 : index
// CHECK:           %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<256x256xf16> -> index
// CHECK:           %[[T3:.*]] = arith.index_cast %[[PTR]] : index to i64
// CHECK:           %[[T4:.*]] = xegpu.create_nd_tdesc %[[T3]], shape : [256, %[[C128]]], strides : [%c128, 1]
// CHECK-SAME:        : i64 -> !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
// CHECK:           %{{.*}} = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}}) -> (vector<8x16xf32>) {
// CHECK:             %[[T7:.*]] = arith.shrui %[[K]], %[[C1]] : index
// CHECK-NEXT:        %[[T8:.*]] = xegpu.load_nd %[[T4]][%{{.*}}, %[[T7]]]
// CHECK-SAME:          <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> :
// CHECK-SAME:          !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<16x8xi32>
// CHECK-NEXT:        %{{.*}} = vector.bitcast %[[T8]] {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
// CHECK-SAME:          : vector<16x8xi32> to vector<16x16xf16>
#a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#b = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
#bt = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>
gpu.module @xevm_module {
gpu.func @gemm_b_transpose(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #a>
  %1 = xegpu.load_nd %0[%c0, %c0]  { layout_result_0 = #a } : !xegpu.tensor_desc<8x16xf32, #a> -> vector<8x16xf32>
  %2 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf16> -> !xegpu.tensor_desc<8x16xf16, #a>
  %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16, #b>
  %4 = scf.for %arg3 = %c0 to %c256 step %c16 iter_args(%arg4 = %1) -> (vector<8x16xf32>) {
    %5 = xegpu.load_nd %2[%c0, %arg3] { layout_result_0 = #a } : !xegpu.tensor_desc<8x16xf16, #a> -> vector<8x16xf16>
    %6 = xegpu.load_nd %3[%c0, %arg3]  { layout_result_0 = #b } : !xegpu.tensor_desc<16x16xf16, #b> -> vector<16x16xf16>
    %7 = vector.transpose %6, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %8 = xegpu.dpas %5, %7, %arg4 {layout_result_0 = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    scf.yield %8 : vector<8x16xf32>
  } {layout_result_0 = #a}
  xegpu.store_nd %4, %0[%c0, %c0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @nested_scf(
// CHECK-SAME:     %{{.*}}: memref<256x256xf16>, %[[ARG1:[a-zA-Z0-9]+]]: memref<256x256xf16>, %{{.*}}: memref<256x256xf32>) {
// CHECK:          %[[C128:.*]] = arith.constant 128 : index
// CHECK:          %[[C1:.*]] = arith.constant 1 : index
// CHECK:          %[[C16:.*]] = arith.constant 16 : index
// CHECK:          %[[C256:.*]] = arith.constant 256 : index
// CHECK:          scf.for %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:            %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<256x256xf16> -> index
// CHECK:            %[[T3:.*]] = arith.index_cast %[[PTR]] : index to i64
// CHECK:            %[[T4:.*]] = xegpu.create_nd_tdesc %[[T3]], shape : [256, %[[C128]]], strides : [%[[C128]], 1] : i64
// CHECK-SAME:          -> !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
// CHECK:            %{{.*}} = scf.for %[[K:.*]] = %{{.*}} iter_args(%{{.*}}) -> (vector<8x16xf32>) {
// CHECK:              %[[T7:.*]] = arith.shrui %[[K]], %[[C1]] : index
// CHECK-NEXT:         %[[T8:.*]] = xegpu.load_nd %[[T4]][%{{.*}}, %[[T7]]]  <{layout = #xegpu.layout<
// CHECK-SAME:          lane_layout = [16, 1], lane_data = [1, 1]>}> :
// CHECK-SAME:          !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<16x8xi32>
// CHECK-NEXT:         %{{.*}} = vector.bitcast %[[T8]] {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
// CHECK-SAME:          : vector<16x8xi32> to vector<16x16xf16>
#a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#b = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
#bt = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>
gpu.module @xevm_module {
gpu.func @nested_scf(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  scf.for %arg8 = %c0 to %c256 step %c16 {
    %0 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #a>
    %1 = xegpu.load_nd %0[%arg8, %c0]  { layout_result_0 = #a } : !xegpu.tensor_desc<8x16xf32, #a> -> vector<8x16xf32>
    %2 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf16> -> !xegpu.tensor_desc<8x16xf16, #a>
    %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16, #b>
    %4 = scf.for %arg3 = %c0 to %c256 step %c16 iter_args(%arg4 = %1) -> (vector<8x16xf32>) {
      %5 = xegpu.load_nd %2[%arg8, %arg3] { layout_result_0 = #a } : !xegpu.tensor_desc<8x16xf16, #a> -> vector<8x16xf16>
      %6 = xegpu.load_nd %3[%arg8, %arg3]  { layout_result_0 = #b } : !xegpu.tensor_desc<16x16xf16, #b> -> vector<16x16xf16>
      %7 = vector.transpose %6, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
      %8 = xegpu.dpas %5, %7, %arg4 {layout_result_0 = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      scf.yield %8 : vector<8x16xf32>
    } {layout_result_0 = #a}
    xegpu.store_nd %4, %0[%c0, %c0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  }
  gpu.return
}
}

// -----
// CHECK-LABEL:   gpu.func @large_loads(
// CHECK-SAME:      %{{.*}}: vector<8x16xf16>, %[[ARG1:[a-zA-Z0-9]+]]: memref<256x256xf16>, %{{.*}}: memref<256x256xf32>) {
// CHECK:           %[[C128:.*]] = arith.constant 128 : index
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[CST:.*]] = arith.constant dense<0> : vector<32x16xi32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<256x256xf16> -> index
// CHECK:           %[[T2:.*]] = arith.index_cast %[[PTR]] : index to i64
// CHECK:           %[[T3:.*]] = xegpu.create_nd_tdesc %[[T2]], shape : [256, %[[C128]]], strides : [%[[C128]], 1] : i64
// CHECK-SAME:        -> !xegpu.tensor_desc<32x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
// CHECK:           %{{.*}}:4 = scf.for %[[K:.*]] = %{{.*}} iter_args(%{{.*}}) -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
// CHECK:             %[[T5:.*]] = arith.shrui %[[K]], %[[C1]] : index
// CHECK:             %[[T6:.*]] = xegpu.load_nd %[[T3]][%{{.*}}, %[[T5]]]  <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}>
// CHECK-SAME:          : !xegpu.tensor_desc<32x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<32x8xi32>
// CHECK:             %[[T7:.*]] = vector.insert_strided_slice %[[T6]], %[[CST]]
// CHECK-SAME:          {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, offsets = [0, 0], strides = [1, 1]}
// CHECK-SAME:          : vector<32x8xi32> into vector<32x16xi32>
// CHECK:             %[[T8:.*]] = arith.addi %[[T5]], %[[C8]] : index
// CHECK:             %[[T9:.*]] = xegpu.load_nd %[[T3]][%{{.*}}, %[[T8]]]  <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}>
// CHECK-SAME:          : !xegpu.tensor_desc<32x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<32x8xi32>
// CHECK:             %[[T10:.*]] = vector.insert_strided_slice %[[T9]], %[[T7]]
// CHECK-SAME:          {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, offsets = [0, 8], strides = [1, 1]}
// CHECK-SAME:          : vector<32x8xi32> into vector<32x16xi32>
// CHECK:             %{{.*}} = vector.bitcast %[[T10]] {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
// CHECK-SAME:          : vector<32x16xi32> to vector<32x32xf16>
#a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#b = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
#bt = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>
gpu.module @xevm_module {
gpu.func @large_loads(%arg0: vector<8x16xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #a>
  %1 = xegpu.load_nd %0[%c0, %c0]  { layout_result_0 = #a } : !xegpu.tensor_desc<8x16xf32, #a> -> vector<8x16xf32>
  %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16> -> !xegpu.tensor_desc<32x32xf16, #b>
  %4:4 = scf.for %arg3 = %c0 to %c256 step %c32 iter_args(%arg4 = %1, %arg5 = %1, %arg6 = %1, %arg7 = %1)
    -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
    %6 = xegpu.load_nd %3[%c0, %arg3]  { layout_result_0 = #b } : !xegpu.tensor_desc<32x32xf16, #b> -> vector<32x32xf16>
    %7 = vector.extract_strided_slice %6 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x32xf16> to vector<16x16xf16>
    %8 = vector.extract_strided_slice %6 {offsets = [0, 16], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x32xf16> to vector<16x16xf16>
    %9 = vector.extract_strided_slice %6 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x32xf16> to vector<16x16xf16>
    %10 = vector.extract_strided_slice %6 {offsets = [16, 16], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x32xf16> to vector<16x16xf16>
    %11 = vector.transpose %7, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %12 = vector.transpose %8, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %13 = vector.transpose %9, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %14 = vector.transpose %10, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %15 = xegpu.dpas %arg0, %11, %arg4 {layout_result_0 = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %16 = xegpu.dpas %arg0, %12, %arg5 {layout_result_0 = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %17 = xegpu.dpas %arg0, %13, %arg6 {layout_result_0 = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %18 = xegpu.dpas %arg0, %14, %arg7 {layout_result_0 = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    scf.yield %15, %16, %17, %18 : vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
  } {layout_result_0 = #a, layout_result_1 = #a, layout_result_2 = #a, layout_result_3 = #a}
  xegpu.store_nd %4#0, %0[%c0, %c0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  xegpu.store_nd %4#1, %0[%c0, %c16]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  xegpu.store_nd %4#2, %0[%c16, %c0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  xegpu.store_nd %4#3, %0[%c16, %c16]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  gpu.return
}
}

// -----
// CHECK-LABEL:  gpu.func @array_length(
// CHECK-SAME:      %{{.*}}: vector<8x16xf16>, %[[ARG1:[a-zA-Z0-9]+]]: memref<256x256xf16>, %arg2: memref<256x256xf32>) {
// CHECK:           %[[C128:.*]] = arith.constant 128 : index
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<256x256xf16> -> index
// CHECK:           %[[T2:.*]] = arith.index_cast %[[PTR]] : index to i64
// CHECK:           %[[T3:.*]] = xegpu.create_nd_tdesc %[[T2]], shape : [256, %[[C128]]], strides : [%[[C128]], 1] : i64 ->
// CHECK-SAME:        !xegpu.tensor_desc<32x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
// CHECK:           %{{.*}}:4 = scf.for %[[K:.*]] = %{{.*}} iter_args(%{{.*}}) -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
// CHECK:             %[[T5:.*]] = arith.shrui %[[K]], %[[C1]] : index
// CHECK:             %[[T6:.*]] = xegpu.load_nd %[[T3]][%{{.*}}, %[[T5]]]  <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}>
// CHECK-SAME:          : !xegpu.tensor_desc<32x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<32x8xi32>
// CHECK:             %[[T7:.*]] = vector.bitcast %[[T6]] {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
// CHECK-SAME:          : vector<32x8xi32> to vector<32x16xf16>
// CHECK:             %[[T8:.*]] = arith.addi %[[T5]], %[[C8]] : index
// CHECK:             %[[T9:.*]] = xegpu.load_nd %[[T3]][%{{.*}}, %[[T8]]]  <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}>
// CHECK-SAME:          : !xegpu.tensor_desc<32x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<32x8xi32>
// CHECK:             %[[T10:.*]] = vector.bitcast %[[T9]] {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
// CHECK-SAME:          : vector<32x8xi32> to vector<32x16xf16>
#a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#b = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
#bt = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>
gpu.module @xevm_module {
gpu.func @array_length(%arg0: vector<8x16xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #a>
  %1 = xegpu.load_nd %0[%c0, %c0]  { layout = #a } : !xegpu.tensor_desc<8x16xf32, #a> -> vector<8x16xf32>
  %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16>
    -> !xegpu.tensor_desc<32x16xf16, #b, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %4:4 = scf.for %arg3 = %c0 to %c256 step %c32 iter_args(%arg4 = %1, %arg5 = %1, %arg6 = %1, %arg7 = %1)
    -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
    %6 = xegpu.load_nd %3[%c0, %arg3]  { layout = #b }
      : !xegpu.tensor_desc<32x16xf16, #b, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x16xf16>
    %19 = vector.extract %6[0] { layout_result_0 = #b } : vector<32x16xf16> from vector<2x32x16xf16>
    %20 = vector.extract %6[1] { layout_result_0 = #b } : vector<32x16xf16> from vector<2x32x16xf16>
    %7 = vector.extract_strided_slice %19 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x16xf16> to vector<16x16xf16>
    %8 = vector.extract_strided_slice %19 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x16xf16> to vector<16x16xf16>
    %9 = vector.extract_strided_slice %20 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x16xf16> to vector<16x16xf16>
    %10 = vector.extract_strided_slice %20 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1], layout_result_0 = #b }
      : vector<32x16xf16> to vector<16x16xf16>
    %11 = vector.transpose %7, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %12 = vector.transpose %8, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %13 = vector.transpose %9, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %14 = vector.transpose %10, [1, 0]  { layout_result_0 = #bt } : vector<16x16xf16> to vector<16x16xf16>
    %15 = xegpu.dpas %arg0, %11, %arg4 {layout = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %16 = xegpu.dpas %arg0, %12, %arg5 {layout = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %17 = xegpu.dpas %arg0, %13, %arg6 {layout = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %18 = xegpu.dpas %arg0, %14, %arg7 {layout = #a} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    scf.yield %15, %16, %17, %18 : vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
  } {layout_result_0 = #a, layout_result_1 = #a, layout_result_2 = #a, layout_result_3 = #a}
  xegpu.store_nd %4#0, %0[%c0, %c0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  xegpu.store_nd %4#1, %0[%c0, %c16]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  xegpu.store_nd %4#2, %0[%c16, %c0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  xegpu.store_nd %4#3, %0[%c16, %c16]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #a>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @vector_reduce_2d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<4x16xf32>, %[[ARG2:[0-9a-zA-Z]+]]: memref<256xf32>) {
// CHECK:      %[[ACC_VEC:.*]] = arith.constant dense<1.000000e+00> : vector<16xf32>
// CHECK:      %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<4x16xf32> -> !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK:      %[[LOADED:.*]] = xegpu.load_nd %[[TDESC]][0, 0]  : !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<4x16xf32>
// CHECK:      %[[LOADED_REDUCED:.*]] = vector.multi_reduction <add>, %[[LOADED]], %[[ACC_VEC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} [0] : vector<4x16xf32> to vector<16xf32>
// CHECK:      %[[LOADED_REDUCED_FOR_CROSS:.*]] = vector.reduction <add>, %[[LOADED_REDUCED]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>} : vector<16xf32> into f32
gpu.module @xevm_test {
  gpu.func @vector_reduce_2d(%src: memref<4x16xf32>, %dst: memref<256xf32>) {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>} 1.0 : f32
    %tdesc = xegpu.create_nd_tdesc %src : memref<4x16xf32>
      -> !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0]
      : !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<4x16xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst
     {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>}
     [0, 1] : vector<4x16xf32> to f32
    %reduce_bcast = vector.broadcast %reduce
     {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}
     : f32 to vector<16xf32>

    %offset = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<0> : vector<16xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<1> : vector<16xi1>

    xegpu.store %reduce_bcast, %dst[%offset], %mask {layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} : vector<16xf32>, memref<256xf32>, vector<16xindex>, vector<16xi1>
    gpu.return
  }
}
