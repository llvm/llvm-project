// RUN: mlir-opt --test-xegpu-unrolling-patterns -split-input-file %s | FileCheck %s

gpu.module @xevm_test {

  // CHECK-LABEL: create_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK: [[cast:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32>
  // CHECK-SAME: to !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {__xegpu_blocking_tile_shape__ = array<i64: 8, 16>, __xegpu_blocking_unpack__}
  gpu.func @create_nd_tdesc(%src: memref<24x32xf32>) -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
  }

//-----
  // CHECK-LABEL: load_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<256x318xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<256x318xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[ld:%.+]] = xegpu.load_nd {{.*}}[{{.*}}]  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  // CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  gpu.func @load_nd(%src: memref<256x318xf32>) -> vector<24x32xf32> {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x318xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %ld = xegpu.load_nd %tdesc[8, 16]: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    gpu.return %ld : vector<24x32xf32>
  }

//-----
  // CHECK-LABEL: load_nd_store_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<256x318xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<256x318xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[data:%.+]] = xegpu.load_nd {{.*}}[{{.*}}]  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  // CHECK-COUNT-6: xegpu.store_nd {{.*}}[{{.*}}] : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @load_nd_store_nd(%src: memref<256x318xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x318xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %ld = xegpu.load_nd %tdesc[8, 16]: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    xegpu.store_nd %ld, %tdesc[0, 0] : vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

//-----
  // CHECK-LABEL: prefetch_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.prefetch_nd {{.*}}[{{.*}}] : !xegpu.tensor_desc<8x16xf32>
  gpu.func @prefetch_nd_tdesc(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    xegpu.prefetch_nd %tdesc[8, 16] : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

//-----

  // CHECK-LABEL: load_nd_offsets_at_both_places
  // CHECK-COUNT-2: builtin.unrealized_conversion_cast
  gpu.func @load_nd_offsets_at_both_places(%src: memref<256x318xf32>) -> vector<24x32xf32> {
    %tdesc = xegpu.create_nd_tdesc %src[16, 8] : memref<256x318xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %ld = xegpu.load_nd %tdesc[8, 16]: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    gpu.return %ld : vector<24x32xf32>
  }
}
