// RUN: mlir-opt --test-xegpu-sink-elementwise-conversions -allow-unregistered-dialect --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @convert_layout_bridge_input_mismatch
// CHECK:         %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<32x32xf16>
// CHECK-NEXT:    %[[BRIDGE:.*]] = xegpu.convert_layout %[[V0]]
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

// -----

// CHECK-LABEL: func.func @sink_conversion_out_of_elementwise
// CHECK-DAG:     %[[MASK:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<16x64xi1>
// CHECK-DAG:     %[[DATA:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<16x64xf32>
// CHECK-DAG:     %[[PAD:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} dense<0xFF800000> : vector<16x64xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[DATA]], %[[PAD]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<16x64xi1>, vector<16x64xf32>
// CHECK:         %[[CVT:.*]] = xegpu.convert_layout %[[SEL]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [8, 16]>, target_layout = #xegpu.layout<inst_data = [1, 16]>}>
// CHECK-SAME:      : vector<16x64xf32>
// CHECK:         %{{.*}} = vector.multi_reduction <maximumf>, %[[CVT]], %{{.*}}
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test_sink_conversion {
func.func @sink_conversion_out_of_elementwise() {
  %mask = "some_op"() {
    layout_result_0 = #xegpu.layout<inst_data = [8, 16]>
  } : () -> vector<16x64xi1>
  %cmask = xegpu.convert_layout %mask <{input_layout = #xegpu.layout<inst_data = [8, 16]>,
                                        target_layout = #xegpu.layout<inst_data = [1, 16]>}> : vector<16x64xi1>
  %data = "some_op"() {
    layout_result_0 = #xegpu.layout<inst_data = [8, 16]>
  } : () -> vector<16x64xf32>
  %cdata = xegpu.convert_layout %data <{input_layout = #xegpu.layout<inst_data = [8, 16]>,
                                        target_layout = #xegpu.layout<inst_data = [1, 16]>}> : vector<16x64xf32>
  %pad = arith.constant {
    layout_result_0 = #xegpu.layout<inst_data = [1, 16]>
  } dense<0xFF800000> : vector<16x64xf32>
  %acc = arith.constant {
    layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 16]>, dims = [1]>
  } dense<0xFF800000> : vector<16xf32>

  %sel = arith.select %cmask, %cdata, %pad {
    layout_result_0 = #xegpu.layout<inst_data = [1, 16]>
  } : vector<16x64xi1>, vector<16x64xf32>
  %red = vector.multi_reduction <maximumf>, %sel, %acc {
    layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 16]>, dims = [1]>
  } [1] : vector<16x64xf32> to vector<16xf32>
  return
}
}

// -----

// CHECK-LABEL: func.func @conflicting_operands_disagree_not_sunk
// CHECK:         %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<32x32xf16>
// CHECK:         %[[CVT0:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [8, 16]>, target_layout = #xegpu.layout<inst_data = [16, 16]>}>
// CHECK:         %[[V1:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [32, 16]>} : () -> vector<32x32xf16>
// CHECK:         %[[CVT1:.*]] = xegpu.convert_layout %[[V1]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [32, 16]>, target_layout = #xegpu.layout<inst_data = [16, 16]>}>
// CHECK:         %{{.*}} = arith.addf %[[CVT0]], %[[CVT1]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} : vector<32x32xf16>
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test_no_sink_on_disagreement {
func.func @conflicting_operands_disagree_not_sunk() {
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : () -> vector<32x32xf16>
  %c0 = xegpu.convert_layout %0
     <{input_layout = #xegpu.layout<inst_data = [8, 16]>,
       target_layout = #xegpu.layout<inst_data = [16, 16]>}>
     : vector<32x32xf16>
  %1 = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [32, 16]>} : () -> vector<32x32xf16>
  %c1 = xegpu.convert_layout %1
     <{input_layout = #xegpu.layout<inst_data = [32, 16]>,
       target_layout = #xegpu.layout<inst_data = [16, 16]>}>
     : vector<32x32xf16>
  %2 = arith.addf %c0, %c1 {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} : vector<32x32xf16>
  return
}
}

// -----

// CHECK-LABEL: func.func @finer_operands_not_sunk
// CHECK:         %[[V0:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<16x64xf32>
// CHECK:         %[[CVT0:.*]] = xegpu.convert_layout %[[V0]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [1, 16]>, target_layout = #xegpu.layout<inst_data = [8, 16]>}>
// CHECK:         %[[V1:.*]] = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<16x64xf32>
// CHECK:         %[[CVT1:.*]] = xegpu.convert_layout %[[V1]]
// CHECK-SAME:      <{input_layout = #xegpu.layout<inst_data = [1, 16]>, target_layout = #xegpu.layout<inst_data = [8, 16]>}>
// CHECK:         %{{.*}} = arith.addf %[[CVT0]], %[[CVT1]]
// CHECK-SAME:      {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<16x64xf32>
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test_no_sink_to_finer {
func.func @finer_operands_not_sunk() {
  %a = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<16x64xf32>
  %ca = xegpu.convert_layout %a
     <{input_layout = #xegpu.layout<inst_data = [1, 16]>,
       target_layout = #xegpu.layout<inst_data = [8, 16]>}>
     : vector<16x64xf32>
  %b = "some_op"() {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} : () -> vector<16x64xf32>
  %cb = xegpu.convert_layout %b
     <{input_layout = #xegpu.layout<inst_data = [1, 16]>,
       target_layout = #xegpu.layout<inst_data = [8, 16]>}>
     : vector<16x64xf32>
  %r = arith.addf %ca, %cb {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>}
    : vector<16x64xf32>
  return
}
}
