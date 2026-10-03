// RUN: mlir-opt  --xevm-attach-target='module=xevm_* chip=cri' \
// RUN: --test-xegpu-sg-to-lane-distribute --split-input-file %s | FileCheck %s

// -----
// Fully broadcast source to one row per lane, the layout `xegpu.dpas_mx` wants
// for its scale operands.
// Handled by SgToLaneConvertLayoutBroadcastExtract.
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @broadcast_to_row_per_lane
// CHECK:         %[[SRC:.*]] = "test.some_op"()
// CHECK:         %[[FLAT:.*]] = vector.shape_cast %[[SRC]] : vector<8x1xf8E8M0FNU> to vector<8xf8E8M0FNU>
// CHECK:         %[[LANE:.*]] = gpu.lane_id
// CHECK:         %[[ROWS:.*]] = arith.constant 8 : index
// CHECK:         %[[ROW:.*]] = arith.remui %[[LANE]], %[[ROWS]] : index
// CHECK:         %[[ELEM:.*]] = vector.extract %[[FLAT]][%[[ROW]]] : f8E8M0FNU from vector<8xf8E8M0FNU>
// CHECK:         vector.from_elements %[[ELEM]] : vector<1x1xf8E8M0FNU>
gpu.func @broadcast_to_row_per_lane() {
  %src = "test.some_op"() : () -> vector<8x1xf8E8M0FNU>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>,
      target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>
    }> : vector<8x1xf8E8M0FNU>
  gpu.return
}
}

// -----
// Source broadcast over two lane groups to one row per lane.
// Handled by SgToLaneConvertLayoutPartialBroadcastExtractShuffle.
//
// The source `test.some_op` is left undistributed by the pass, hence the cast to
// the distributed input type; a real, distributed producer needs no cast.
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @lane_group_columns_to_row_per_lane
// CHECK:         %[[SRC:.*]] = "test.some_op"()
// CHECK:         %[[DIST:.*]] = builtin.unrealized_conversion_cast %[[SRC]] : vector<8x2xf8E8M0FNU> to vector<8x1xf8E8M0FNU>
// CHECK:         %[[FLAT:.*]] = vector.shape_cast %[[DIST]] : vector<8x1xf8E8M0FNU> to vector<8xf8E8M0FNU>
// CHECK:         %[[LANE:.*]] = gpu.lane_id
// CHECK:         %[[ROWS:.*]] = arith.constant 8 : index
// CHECK:         %[[ROW:.*]] = arith.remui %[[LANE]], %[[ROWS]] : index
// CHECK:         %[[OWN:.*]] = vector.extract %[[FLAT]][%[[ROW]]] : f8E8M0FNU from vector<8xf8E8M0FNU>
// CHECK:         %[[WIDTH:.*]] = arith.constant 16 : i32
// CHECK:         %[[ROWI:.*]] = arith.index_cast %[[ROW]] : index to i32
// CHECK:         %[[STRIDE:.*]] = arith.constant 8 : i32
// CHECK:         %[[OTHER:.*]] = arith.addi %[[ROWI]], %[[STRIDE]] : i32
// CHECK:         %[[COL0:.*]], %{{.*}} = gpu.shuffle idx %[[OWN]], %[[ROWI]], %[[WIDTH]] : f8E8M0FNU
// CHECK:         %[[COL1:.*]], %{{.*}} = gpu.shuffle idx %[[OWN]], %[[OTHER]], %[[WIDTH]] : f8E8M0FNU
// CHECK:         vector.from_elements %[[COL0]], %[[COL1]] : vector<1x2xf8E8M0FNU>
gpu.func @lane_group_columns_to_row_per_lane() {
  %src = "test.some_op"() : () -> vector<8x2xf8E8M0FNU>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>,
      target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>
    }> : vector<8x2xf8E8M0FNU>
  gpu.return
}
}

// -----
// Fully broadcast source to one column per lane group.
// Handled by SgToLaneConvertLayoutDeinterleaveSelect.
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @broadcast_to_column_per_lane_group
// CHECK:         %[[SRC:.*]] = "test.some_op"()
// CHECK:         %[[FLAT:.*]] = vector.shape_cast %[[SRC]] : vector<8x2xbf16> to vector<16xbf16>
// CHECK:         %[[EVEN:.*]], %[[ODD:.*]] = vector.deinterleave %[[FLAT]] : vector<16xbf16> -> vector<8xbf16>
// CHECK:         %[[LANE:.*]] = gpu.lane_id
// CHECK:         %[[STRIDE:.*]] = arith.constant 8 : index
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:         %[[GROUP:.*]] = arith.divui %[[LANE]], %[[STRIDE]] : index
// CHECK:         %[[FIRST:.*]] = arith.cmpi eq, %[[GROUP]], %[[ZERO]] : index
// CHECK:         %[[SEL:.*]] = arith.select %[[FIRST]], %[[EVEN]], %[[ODD]] : vector<8xbf16>
// CHECK:         vector.shape_cast %[[SEL]] : vector<8xbf16> to vector<8x1xbf16>
gpu.func @broadcast_to_column_per_lane_group() {
  %src = "test.some_op"() : () -> vector<8x2xbf16>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>,
      target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>
    }> : vector<8x2xbf16>
  gpu.return
}
}
