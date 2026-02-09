// RUN: mlir-opt --test-xegpu-unrolling-patterns -split-input-file %s | FileCheck %s

gpu.module @test {

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

  // CHECK-LABEL: create_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK: [[cast:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-SAME: !xegpu.tensor_desc<16xf32>
  // CHECK-SAME: to !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {__xegpu_blocking_tile_shape__ = array<i64: 16>, __xegpu_blocking_unpack__}
  gpu.func @create_nd_tdesc_1d(%src: memref<64xf32>) -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64xf32> -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
  }

  //-----

  // CHECK-LABEL: prefetch_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.prefetch_nd {{.*}} : !xegpu.tensor_desc<8x16xf32>
  gpu.func @prefetch_nd_tdesc(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    xegpu.prefetch_nd %tdesc[0, 0] : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: prefetch_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.prefetch_nd {{.*}} : !xegpu.tensor_desc<16xf32>
  gpu.func @prefetch_nd_tdesc_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    xegpu.prefetch_nd %tdesc[0] : !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

  //-----
  // CHECK-LABEL: load_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[ld:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  // CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  gpu.func @load_nd(%src: memref<24x32xf32>) -> vector<24x32xf32> {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %ld = xegpu.load_nd %tdesc[0, 0]: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    gpu.return %ld : vector<24x32xf32>
  }

  //-----

  // CHECK-LABEL: load_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: [[ld:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
  // CHECK-COUNT-4: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<16xf32> into vector<64xf32>
  gpu.func @load_nd_1d(%src: memref<64xf32>) -> vector<64xf32> {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = xegpu.load_nd %tdesc[0]: !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>> -> vector<64xf32>
    gpu.return %data : vector<64xf32>
  }

  //-----

  // CHECK-LABEL: store_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.store_nd {{.*}}  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @store_nd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    xegpu.store_nd %data, %tdesc[0, 0]: vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: store_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.store_nd {{.*}}  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  gpu.func @store_nd_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = arith.constant dense<9.0> : vector<64xf32>
    xegpu.store_nd %data, %tdesc[0]: vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: createNd_loadNd_storeNd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  //CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-COUNT-6: [[data:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  //CHECK: [[add:%.+]] = arith.addf {{.*}} : vector<24x32xf32>
  //CHECK-COUNT-6: [[extract:%.+]] = vector.extract_strided_slice {{.*}} : vector<24x32xf32> to vector<8x16xf32>
  //CHECK-COUNT-6: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @createNd_loadNd_storeNd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    %ld = xegpu.load_nd %tdesc[0, 0]: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    %add = arith.addf %data, %ld : vector<24x32xf32>
    xegpu.store_nd %add, %tdesc[0, 0]: vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: dpas
  // CHECK-SAME: [[arg0:%.+]]: vector<32x32xf16>, [[arg1:%.+]]: vector<32x32xf16>
  //CHECK-COUNT-8: [[extract1:%.+]] = vector.extract_strided_slice [[arg0]] {{.*}} : vector<32x32xf16> to vector<8x16xf16>
  //CHECK-COUNT-4: [[extract2:%.+]] = vector.extract_strided_slice [[arg1]] {{.*}} : vector<32x32xf16> to vector<16x16xf16>
  //CHECK-COUNT-16: [[dpas:%.+]] = xegpu.dpas {{.*}} -> vector<8x16xf32>
  //CHECK-COUNT-8: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<32x32xf32>
  gpu.func @dpas(%a: vector<32x32xf16>, %b: vector<32x32xf16>) -> vector<32x32xf32> {
    %c = xegpu.dpas %a, %b : vector<32x32xf16>, vector<32x32xf16> -> vector<32x32xf32>
    gpu.return %c : vector<32x32xf32>
  }
}

