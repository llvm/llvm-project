// RUN: mlir-opt --test-xegpu-resolve-layout-conflicts -allow-unregistered-dialect -split-input-file %s | FileCheck %s

#inst_data_8x16 = #xegpu.layout<inst_data = [8, 16]>
#inst_data_16x16 = #xegpu.layout<inst_data = [16, 16]>
#inst_data_32x16 = #xegpu.layout<inst_data = [32, 16]>
#inst_data_16 = #xegpu.layout<inst_data = [16]>
#inst_data_32 = #xegpu.layout<inst_data = [32]>
#inst_data_1x2x16 = #xegpu.layout<inst_data = [1, 2, 16]>
#inst_data_1x32 = #xegpu.layout<inst_data = [1, 32]>
gpu.module @test {

// CHECK-LABEL:   func.func @load_nd_with_conflicting_tensor_desc
// CHECK:           %{{.*}} = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK-NEXT:      %{{.*}} = xegpu.load_nd %[[T1]][%{{.*}}, %{{.*}}] <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<16x16xf16>
func.func @load_nd_with_conflicting_tensor_desc(%arg0: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<64x64xf16>
    -> !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>
  %1 = xegpu.load_nd %0 [%c0, %c0] {layout = #inst_data_8x16} : !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>
    -> vector<16x16xf16>
  xegpu.prefetch_nd %0 [%c0, %c0] {layout = #inst_data_16x16} : !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>
  return
}

// CHECK-LABEL:   func.func @multiple_tensor_desc_conflicts
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK-NEXT:      %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      %[[T2:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [32, 16]>>
// CHECK-NEXT:      %{{.*}} = xegpu.load_nd %[[T0]][%[[C0]], %[[C0]]] <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<32x16xf16>
// CHECK-NEXT:      %{{.*}} = xegpu.load_nd %[[T2]][%[[C0]], %[[C0]]] <{layout = #xegpu.layout<inst_data = [32, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [32, 16]>> -> vector<32x16xf16>
// CHECK-NEXT:      xegpu.prefetch_nd %[[T1]][%[[C0]], %[[C0]]] <{layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [16, 16]>>
func.func @multiple_tensor_desc_conflicts(%arg0: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %tdesc1 = xegpu.create_nd_tdesc %arg0 : memref<64x64xf16>
    -> !xegpu.tensor_desc<32x16xf16, #inst_data_8x16>
  %load1 = xegpu.load_nd %tdesc1 [%c0, %c0] {layout = #inst_data_8x16} : !xegpu.tensor_desc<32x16xf16, #inst_data_8x16>
    -> vector<32x16xf16>
  %load2 = xegpu.load_nd %tdesc1 [%c0, %c0] {layout = #inst_data_32x16} : !xegpu.tensor_desc<32x16xf16, #inst_data_8x16>
    -> vector<32x16xf16>
  xegpu.prefetch_nd %tdesc1 [%c0, %c0] {layout = #inst_data_16x16} : !xegpu.tensor_desc<32x16xf16, #inst_data_8x16>
  return
}

// CHECK-LABEL:   func.func @load_nd_with_conflicting_tensor_desc_in_loop
// CHECK:           %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK-NEXT:      %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %[[T0]])
// CHECK-SAME:        -> (vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>) {
// CHECK-NEXT:        %{{.*}} = xegpu.load_nd %[[T1]][%{{.*}}] <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME:          !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<16x16xf16>
// CHECK:             scf.yield %{{.*}}, %{{.*}} : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK:           xegpu.prefetch_nd %[[T0]][%{{.*}}] <{layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      return
func.func @load_nd_with_conflicting_tensor_desc_in_loop(%arg0: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant {layout_result_0 = #inst_data_8x16} dense<0.0> : vector<16x16xf16>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<64x64xf16>
    -> !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>
  %1:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst, %tdesc = %0)
    -> (vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>) {
    %2 = xegpu.load_nd %tdesc [%c0, %c0] {layout = #inst_data_8x16} : !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>
      -> vector<16x16xf16>
    %3 = arith.addf %acc, %2 {layout_result_0 = #inst_data_8x16} : vector<16x16xf16>
    scf.yield %3, %tdesc : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>
  } {layout_result_0 = #inst_data_8x16}
  xegpu.prefetch_nd %0 [%c0, %c0] {layout = #inst_data_16x16} : !xegpu.tensor_desc<16x16xf16, #inst_data_16x16>
  return
}


// CHECK-LABEL: func.func @elementwise_conflict
// CHECK-DAG:     %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<32x32xf16>
// CHECK-DAG:     %[[V1:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [32, 16]>} : () -> vector<32x32xf16>
// CHECK-DAG:     %[[CVT:.*]] = xegpu.convert_layout %[[V1]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [32, 16]>, target_layout = #xegpu.layout<inst_data = [8, 16]>}>
// CHECK-SAME:      : vector<32x32xf16>
// CHECK:         %[[ADD:.*]] = arith.addf %[[V0]], %[[CVT]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<32x32xf16>
// CHECK:         return
func.func @elementwise_conflict() {
  %0 = "some_op"() {layout_result_0 = #inst_data_8x16} : () -> vector<32x32xf16>
  %1 = "some_op"() {layout_result_0 = #inst_data_32x16} : () -> vector<32x32xf16>
  %2 = arith.addf %0, %1 {layout_result_0 = #inst_data_8x16} : vector<32x32xf16>
  return
}

// CHECK-LABEL: func.func @elementwise_conflict_uniform
// CHECK-DAG:     %[[V0:.*]] = "some_op"() : () -> vector<2xf16>
// CHECK-DAG:     %[[V1:.*]] = "some_op"() : () -> vector<2xf16>
// CHECK:         %[[ADD:.*]] = arith.addf %[[V0]], %[[V1]] : vector<2xf16>
// CHECK:         return
func.func @elementwise_conflict_uniform() {
  %0 = "some_op"() : () -> vector<2xf16>
  %1 = "some_op"() : () -> vector<2xf16>
  %non_tensor_data_vec = arith.addf %0, %1 : vector<2xf16>
  return
}

// CHECK-LABEL: func.func @broadcast_source_conflict
// CHECK:         %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [16]>} : () -> vector<16xf16>
// CHECK:         %[[CVT:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [16]>, target_layout = #xegpu.slice<#xegpu.layout<inst_data = [16, 16]>, dims = [0]>}>
// CHECK-SAME:      : vector<16xf16>
// CHECK:         %[[BC:.*]] = vector.broadcast %[[CVT]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} : vector<16xf16> to vector<16x16xf16>
// CHECK:         return
func.func @broadcast_source_conflict() {
  %0 = "some_op"() {layout_result_0 = #inst_data_16} : () -> vector<16xf16>
  %1 = vector.broadcast %0 {layout_result_0 = #inst_data_16x16} : vector<16xf16> to vector<16x16xf16>
  return
}

// CHECK-LABEL: func.func @shapecast_source_conflict
// CHECK:         %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 2, 16]>} : () -> vector<2x4x32xf16>
// CHECK:         %[[CVT:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [1, 2, 16]>, target_layout = #xegpu.layout<inst_data = [1, 1, 32]>}>
// CHECK-SAME:      : vector<2x4x32xf16>
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[CVT]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [1, 32]>} : vector<2x4x32xf16> to vector<1x256xf16>
// CHECK:         return
func.func @shapecast_source_conflict() {
  %0 = "some_op"() {layout_result_0 = #inst_data_1x2x16} : () -> vector<2x4x32xf16>
  %1 = vector.shape_cast %0 {layout_result_0 = #inst_data_1x32}  : vector<2x4x32xf16> to vector<1x256xf16>
  return
}

// CHECK-LABEL: func.func @bitcast_source_conflict
// CHECK:         %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<32x16xf32>
// CHECK:         %[[CVT:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [1, 16]>, target_layout = #xegpu.layout<inst_data = [16, 16]>}>
// CHECK-SAME:      : vector<32x16xf32>
// CHECK:         %[[BC:.*]] = vector.bitcast %[[CVT]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [16, 32]>} : vector<32x16xf32> to vector<32x32xf16>
// CHECK:         return
func.func @bitcast_source_conflict() {
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<32x16xf32>
  %1 = vector.bitcast %0 {layout_result_0 = #xegpu.layout<inst_data = [16, 32]>} : vector<32x16xf32> to vector<32x32xf16>
  return
}

// CHECK-LABEL: func.func @multireduction_source_conflict
// CHECK-DAG:     %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [32, 16]>} : () -> vector<32x32xf16>
// CHECK-DAG:     %[[CVT0:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [32, 16]>, target_layout = #xegpu.layout<inst_data = [16, 16]>}>
// CHECK-SAME:      : vector<32x32xf16>
// CHECK-DAG:     %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [32]>}
// CHECK-SAME:      dense<0.000000e+00> : vector<32xf16>
// CHECK-DAG:     %[[CVT1:.*]] = xegpu.convert_layout %[[CST]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [32]>, target_layout = #xegpu.slice<#xegpu.layout<inst_data = [16, 16]>, dims = [0]>}>
// CHECK-SAME:      : vector<32xf16>
// CHECK:         %[[MR:.*]] = vector.multi_reduction <add>, %[[CVT0]], %[[CVT1]]
// CHECK-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [16, 16]>, dims = [0]>}
// CHECK-SAME:      [0] : vector<32x32xf16> to vector<32xf16>
// CHECK:         return
func.func @multireduction_source_conflict() {
  %0 = "some_op"() {layout_result_0 = #inst_data_32x16} : () -> vector<32x32xf16>
  %acc = arith.constant {layout_result_0 = #inst_data_32} dense<0.0> : vector<32xf16>
  %1 = vector.multi_reduction <add>, %0, %acc
    {layout_result_0 = #xegpu.slice<#inst_data_16x16, dims = [0]>}
    [0] : vector<32x32xf16> to vector<32xf16>
  return
}

// CHECK-LABEL: func.func @insert_strided_slice_source_conflict
// CHECK-DAG:     %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<16x16xf16>
// CHECK-DAG:     %[[CVT:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [1, 16]>, target_layout = #xegpu.layout<inst_data = [16, 16]>}>
// CHECK-SAME:      : vector<16x16xf16>
// CHECK-DAG:     %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [1, 16, 16]>}
// CHECK-SAME:      dense<0.000000e+00> : vector<2x32x32xf16>
// CHECK:         %[[ISS:.*]] = vector.insert_strided_slice %[[CVT]], %[[CST]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [1, 16, 16]>, offsets = [0, 0, 0], strides = [1, 1]}
// CHECK-SAME:      : vector<16x16xf16> into vector<2x32x32xf16>
// CHECK:         return
func.func @insert_strided_slice_source_conflict() {
  %0 = "some_op"()  {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<16x16xf16>
  %1 = arith.constant  { layout_result_0 = #xegpu.layout<inst_data = [1, 16, 16]>}
    dense<0.0> : vector<2x32x32xf16>
  %2 = vector.insert_strided_slice %0, %1 {offsets = [0, 0, 0], strides = [1, 1],
    layout_result_0 = #xegpu.layout<inst_data = [1, 16, 16]>} : vector<16x16xf16> into vector<2x32x32xf16>
  return
}

// CHECK-LABEL: func.func @conflict_inside_loop
// CHECK-DAG:     %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>}
// CHECK-SAME:      dense<0.000000e+00> : vector<16x16xf16>
// CHECK:         %[[FOR:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[CST]]) -> (vector<16x16xf16>) {
// CHECK:           %[[V1:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} : () -> vector<16x16xf16>
// CHECK:           %[[CVT:.*]] = xegpu.convert_layout %[[V1]]
// CHECK-SAME:        <{input_layout = #xegpu.layout<inst_data = [16, 16]>, target_layout = #xegpu.layout<inst_data = [8, 16]>}>
// CHECK-SAME:        : vector<16x16xf16>
// CHECK:           %[[ADD:.*]] = arith.addf %[[ACC]], %[[CVT]]
// CHECK-SAME:        {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<16x16xf16>
// CHECK:           scf.yield %[[ADD]] : vector<16x16xf16>
// CHECK:         }
// CHECK:         return
func.func @conflict_inside_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant {layout_result_0 = #inst_data_8x16} dense<0.0> : vector<16x16xf16>
  %0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> vector<16x16xf16> {
    %1 = "some_op"() {layout_result_0 = #inst_data_16x16} : () -> vector<16x16xf16>
    %2 = arith.addf %acc, %1 {layout_result_0 = #inst_data_8x16} : vector<16x16xf16>
    scf.yield %2 : vector<16x16xf16>
  } {layout_result_0 = #inst_data_8x16}
  return
}

// CHECK-LABEL: func.func @conflict_postop
// CHECK-DAG:     %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>}
// CHECK-SAME:      dense<0.000000e+00> : vector<16x16xf16>
// CHECK:         %[[FOR:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[CST]]) -> (vector<16x16xf16>) {
// CHECK:           %[[V1:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} : () -> vector<16x16xf16>
// CHECK:           %[[ADD:.*]] = arith.addf %[[ACC]], %[[V1]]
// CHECK-SAME:        {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} : vector<16x16xf16>
// CHECK:           scf.yield %[[ADD]] : vector<16x16xf16>
// CHECK:         }
// CHECK:         %[[CVT:.*]] = xegpu.convert_layout %[[FOR]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [16, 16]>, target_layout = #xegpu.layout<inst_data = [8, 16]>}>
// CHECK-SAME:      : vector<16x16xf16>
// CHECK:         %[[EXP:.*]] = math.exp %[[CVT]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<16x16xf16>
// CHECK:         return
func.func @conflict_postop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant {layout_result_0 = #inst_data_16x16} dense<0.0> : vector<16x16xf16>
  %0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> vector<16x16xf16> {
    %1 = "some_op"() {layout_result_0 = #inst_data_16x16} : () -> vector<16x16xf16>
    %2 = arith.addf %acc, %1 {layout_result_0 = #inst_data_16x16} : vector<16x16xf16>
    scf.yield %2 : vector<16x16xf16>
  } {layout_result_0 = #inst_data_16x16}
  %1 = math.exp %0 {layout_result_0 = #inst_data_8x16} : vector<16x16xf16>
  return
}

// CHECK-LABEL: func.func @convert_layout
// CHECK: %[[V0:.*]] = xegpu.convert_layout %[[CST:.*]] <{input_layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [4, 32]>, target_layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [4, 32]>}> : vector<32x128xf32>
// CHECK: %[[V1:.*]] = xegpu.convert_layout %[[V0]]  <{input_layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [4, 32]>, target_layout = #xegpu.layout<sg_layout = [4, 8], sg_data = [8, 16]>}> : vector<32x128xf32>
func.func @convert_layout() {
  %src0 = arith.constant
    {layout_result_0 = #xegpu.layout<sg_layout=[8, 4], sg_data=[4, 32]>}
      dense<0.000000e+00>
     : vector<32x128xf32>
  %src0_cvt = xegpu.convert_layout %src0
    <{input_layout = #xegpu.layout<sg_layout=[8, 4], sg_data=[4, 32]>,
     target_layout = #xegpu.layout<sg_layout=[8, 4], sg_data=[4, 32]>}>
    : vector<32x128xf32>
  %src1 = arith.constant
    {layout_result_0 = #xegpu.layout<sg_layout = [4, 8], sg_data = [8, 16]>}
    dense<1.000000e+00>
    : vector<32x128xf32>
  %dest = arith.addf %src0_cvt, %src1
    {layout_result_0 = #xegpu.layout<sg_layout = [4, 8], sg_data = [8, 16]>}
    : vector<32x128xf32>
  %desc_cvt = xegpu.convert_layout %dest
     <{input_layout = #xegpu.layout<sg_layout=[4, 8], sg_data=[8, 16]>,
      target_layout = #xegpu.layout<sg_layout=[4, 8], sg_data=[8, 16]>}>
     : vector<32x128xf32>
  return
}

// CHECK-LABEL: func.func @extract_source_conflict_with_order
// CHECK-DAG:     %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 1, 16], lane_data = [1, 1, 1, 1], order = [2, 3, 0, 1]>} : () -> vector<2x4x16x32xf16>
// CHECK-DAG:     %[[CVT:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<lane_layout = [1, 1, 1, 16], lane_data = [1, 1, 1, 1], order = [2, 3, 0, 1]>, target_layout = #xegpu.layout<lane_layout = [1, 1, 1, 16], lane_data = [1, 1, 1, 1], order = [2, 3, 1, 0]>}>
// CHECK-SAME:      : vector<2x4x16x32xf16>
// CHECK:         %[[EXT:.*]] = vector.extract %[[CVT]][0, 0]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [0, 1]>}
// CHECK-SAME:      : vector<16x32xf16> from vector<2x4x16x32xf16>
// CHECK:         return %[[EXT]] : vector<16x32xf16>
func.func @extract_source_conflict_with_order() -> vector<16x32xf16> {
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 1, 16], lane_data = [1, 1, 1, 1], order = [2, 3, 0, 1]>} : () -> vector<2x4x16x32xf16>
  %1 = vector.extract %0[0, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [0, 1]>} : vector<16x32xf16> from vector<2x4x16x32xf16>
  return %1 : vector<16x32xf16>
}
}

// -----

// Producer carries a layout that differs from the input_layout declared on a
// downstream xegpu.convert_layout consumer. ResolveLayoutConflicts must insert
// a bridging convert_layout between the producer and the existing
// convert_layout so the consumer's stated input_layout is honored.
// CHECK-LABEL: func.func @convert_layout_bridge_input_mismatch
// CHECK:         %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<32x32xf16>
// CHECK-NEXT:    %[[BRIDGE:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [8, 16]>, target_layout = #xegpu.layout<inst_data = [16, 16]>}>
// CHECK-SAME:      : vector<32x32xf16>
// CHECK-NEXT:    %[[CVT:.*]] = xegpu.convert_layout %[[BRIDGE]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [16, 16]>, target_layout = #xegpu.layout<inst_data = [32, 16]>}>
// CHECK-SAME:      : vector<32x32xf16>
gpu.module @test_convert_layout_bridge {
func.func @convert_layout_bridge_input_mismatch() {
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<32x32xf16>
  %1 = xegpu.convert_layout %0
     <{input_layout = #xegpu.layout<inst_data = [16, 16]>,
       target_layout = #xegpu.layout<inst_data = [32, 16]>}>
     : vector<32x32xf16>
  return
}
}
