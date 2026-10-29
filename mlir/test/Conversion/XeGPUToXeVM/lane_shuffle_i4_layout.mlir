// RUN: mlir-opt --test-xegpu-sg-to-lane-distribute %s | FileCheck %s --check-prefix=DIST
// RUN: mlir-opt --test-xegpu-sg-to-lane-distribute --convert-xegpu-to-xevm %s | FileCheck %s --implicit-check-not='arith.{{(extui|extsi|trunci)}} {{.*}}xi4>' --implicit-check-not=arith.ori --implicit-check-not=xegpu.lane_shuffle

// A contiguous four-nibble lane fragment is unpacked to round-robin order,
// then extended for a vector store. Slice one row from the 8x64 tile so the
// layout conversion produces one shuffle, without unsupported vector inserts.
// Exercise the shuffle produced by layout distribution with a real sub-byte
// block load rather than a fragment argument. The test distribution pass does
// not distribute vector.store, so it materializes its input with a cast.
gpu.module @test [#xevm.target<chip = "cri">] {
  // DIST-LABEL: gpu.func @load_convert_layout_extui
  // DIST: %[[LOAD:.*]] = xegpu.load_nd {{.*}} -> vector<32xi4>
  // DIST: %[[TILE:.*]] = vector.shape_cast %[[LOAD]] : vector<32xi4> to vector<8x4xi4>
  // DIST: %[[ROW:.*]] = vector.extract_strided_slice %[[TILE]] offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : vector<8x4xi4> to vector<1x4xi4>
  // DIST: %[[FLAT:.*]] = vector.shape_cast %[[ROW]] : vector<1x4xi4> to vector<4xi4>
  // DIST: %[[SHUFFLED:.*]] = xegpu.lane_shuffle %[[FLAT]] unpack : vector<4xi4>
  // DIST: %[[SHAPED:.*]] = vector.shape_cast %[[SHUFFLED]] : vector<4xi4> to vector<1x4xi4>
  // DIST: %[[EXT:.*]] = arith.extui %[[SHAPED]] : vector<1x4xi4> to vector<1x4xi32>
  // DIST: %[[STORE:.*]] = builtin.unrealized_conversion_cast %[[EXT]] : vector<1x4xi32> to vector<1x64xi32>
  // DIST: vector.store %[[STORE]]
  // CHECK-LABEL: gpu.func @load_convert_layout_extui
  // CHECK-SAME: %[[SRC:.*]]: memref<8x64xi4>, %[[DST:.*]]: memref<8x64xi32>
  gpu.func @load_convert_layout_extui(%src: memref<8x64xi4>, %dst: memref<8x64xi32>) {
    // CHECK: %[[LOAD:.*]] = xevm.blockload2d
    // CHECK-SAME: elem_size_in_bits = 16 : i32
    // CHECK-SAME: tile_height = 8 : i32, tile_width = 16 : i32
    // CHECK-SAME: -> vector<8xi16>
    // CHECK: %[[NIBBLES:.*]] = vector.bitcast %[[LOAD]] : vector<8xi16> to vector<32xi4>
    // CHECK: %[[TILE:.*]] = vector.shape_cast %[[NIBBLES]] : vector<32xi4> to vector<8x4xi4>
    // CHECK: %[[ROW:.*]] = vector.extract_strided_slice %[[TILE]] offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : vector<8x4xi4> to vector<1x4xi4>
    // CHECK: %[[FLAT:.*]] = vector.shape_cast %[[ROW]] : vector<1x4xi4> to vector<4xi4>
    // CHECK: %[[BYTES:.*]] = vector.bitcast %[[FLAT]] : vector<4xi4> to vector<2xi8>
    // CHECK: %[[LOW:.*]] = arith.andi %[[BYTES]], %{{.*}} : vector<2xi8>
    // CHECK: %[[HIGH:.*]] = arith.shrui %[[BYTES]], %{{.*}} : vector<2xi8>
    // CHECK: %[[WIDE:.*]] = vector.shuffle %[[LOW]], %[[HIGH]] [0, 2, 1, 3] : vector<2xi8>, vector<2xi8>
    // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[WIDE]] : vector<4xi8> to i32
    // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<4xi8>
    // CHECK: %[[SHAPED:.*]] = vector.shape_cast %[[SHUFFLED]] : vector<4xi8> to vector<1x4xi8>
    // CHECK: %[[EXT:.*]] = arith.extui %[[SHAPED]] : vector<1x4xi8> to vector<1x4xi32>
    // CHECK: %[[STORE:.*]] = builtin.unrealized_conversion_cast %[[EXT]] : vector<1x4xi32> to vector<1x64xi32>
    // CHECK: vector.store %[[STORE]], %[[DST]]
    %c0 = arith.constant 0 : index
    %tdesc = xegpu.create_nd_tdesc %src : memref<8x64xi4> -> !xegpu.tensor_desc<8x64xi4>
    %loaded = xegpu.load_nd %tdesc[%c0, %c0]
      <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 4]>}>
      : !xegpu.tensor_desc<8x64xi4> -> vector<8x64xi4>
    %row = vector.extract_strided_slice %loaded offsets = [0, 0], sizes = [1, 64], strides = [1, 1]
      : vector<8x64xi4> to vector<1x64xi4>
    %converted = xegpu.convert_layout %row
      <{input_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 4]>,
        target_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}>
      : vector<1x64xi4>
    %extended = arith.extui %converted : vector<1x64xi4> to vector<1x64xi32>
    %anchor = xegpu.convert_layout %extended
      <{target_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}>
      : vector<1x64xi32>
    vector.store %anchor, %dst[%c0, %c0] : memref<8x64xi32>, vector<1x64xi32>
    gpu.return
  }
}
