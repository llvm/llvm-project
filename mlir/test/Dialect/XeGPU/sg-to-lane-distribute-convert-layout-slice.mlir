// RUN: mlir-opt  --xevm-attach-target='module=xevm_* chip=cri' \
// RUN: --test-xegpu-sg-to-lane-distribute --split-input-file %s | FileCheck %s

// -----
// The input layout leaves effective lane_layout [1, 1], so the value is fully
// broadcast: every lane holds all 8 rows. The target spreads the 8 rows over 8
// lanes, one row each, which is the layout `xegpu.dpas_mx` wants for its scale
// operands. Lane `l` keeps row `l % 8` and no data crosses lanes, so a single
// extract suffices.
//
// The source is flattened to rank 1 first because `xegpu-vector-linearize`
// cannot linearize a `vector.extract` with a dynamic position out of a rank-2
// value.
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
// The input layout leaves effective lane_layout [1, 2], with the distributed
// dimension at lane stride 8, so the value is broadcast over two lane groups:
// lane `l` holds all 8 rows of column `l / 8`. The target wants both columns of
// row `l % 8` in lane `l`, so each lane extracts its own column and shuffles
// the other one in from lane `l ^ 8`, which holds the same row of the other
// column.
//
// Lanes 8..15 receive the two columns in the opposite order, which is unused
// because the target lane_layout [8, 1] only occupies 8 lanes.
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
// The offset and width constants are created as arguments of the same call in
// `gpu::ShuffleOp::build`, so their relative order is up to the host compiler.
// CHECK-DAG:     %[[OFFSET:.*]] = arith.constant 8 : i32
// CHECK-DAG:     %[[WIDTH:.*]] = arith.constant 16 : i32
// CHECK:         %[[PARTNER:.*]], %{{.*}} = gpu.shuffle xor %[[OWN]], %[[OFFSET]], %[[WIDTH]] : f8E8M0FNU
// CHECK:         vector.from_elements %[[OWN]], %[[PARTNER]] : vector<1x2xf8E8M0FNU>
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
// The input layout leaves effective lane_layout [1, 1], so the value is fully
// broadcast. The target leaves effective lane_layout [1, 2], with the
// distributed dimension at lane stride 8: lanes 0..7 keep column 0 and lanes
// 8..15 keep column 1. A column is a stride-2 subset of the row-major value, so
// `vector.deinterleave` separates the two and the lane group selects which one
// the lane keeps.
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
