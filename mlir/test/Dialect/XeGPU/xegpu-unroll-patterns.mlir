// RUN: mlir-opt --test-xegpu-unrolling-patterns -split-input-file %s | FileCheck %s

gpu.module @test {

  // CHECK-LABEL: create_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK: [[cast:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32>, !xegpu.tensor_desc<8x16xf32>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32>, !xegpu.tensor_desc<8x16xf32>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  // CHECK-SAME: to !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {__xegpu_blocking_tile_shape__ = array<i64: 8, 16>, __xegpu_blocking_unpack__}
  gpu.func @create_nd_tdesc(%src: memref<24x32xf32>) -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
  }

  //-----

  // CHECK-LABEL: create_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-2: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK: [[cast:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-SAME: !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>
  // CHECK-SAME: to !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {__xegpu_blocking_tile_shape__ = array<i64: 16>, __xegpu_blocking_unpack__}
  gpu.func @create_nd_tdesc_1d(%src: memref<64xf32>) -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
  }

  //-----

  // CHECK-LABEL: update_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[update:%.+]] = xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<8x16xf32>
  gpu.func @update_nd_tdesc(%src: memref<24x32xf32>) -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %update = xegpu.update_nd_offset %tdesc, [0, 16] : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return %update : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
  }

  //-----

  // CHECK-LABEL: update_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-2: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-2: [[update:%.+]] = xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<16xf32>
  gpu.func @update_nd_tdesc_1d(%src: memref<64xf32>) -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
    %update = xegpu.update_nd_offset %tdesc, [32] : !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return %update : !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
  }

  //-----

  // CHECK-LABEL: prefetch_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.prefetch_nd {{.*}} : !xegpu.tensor_desc<8x16xf32>
  gpu.func @prefetch_nd_tdesc(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    xegpu.prefetch_nd %tdesc : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: prefetch_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-4: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.prefetch_nd {{.*}} : !xegpu.tensor_desc<16xf32>
  gpu.func @prefetch_nd_tdesc_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    xegpu.prefetch_nd %tdesc : !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

  //-----
  // CHECK-LABEL: load_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[ld:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  // CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  gpu.func @load_nd(%src: memref<24x32xf32>) -> vector<24x32xf32> {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %ld = xegpu.load_nd %tdesc: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    gpu.return %ld : vector<24x32xf32>
  }

  //-----

  // CHECK-LABEL: load_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-4: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: [[ld:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
  // CHECK-COUNT-4: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<16xf32> into vector<64xf32>
  gpu.func @load_nd_1d(%src: memref<64xf32>) -> vector<64xf32> {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = xegpu.load_nd %tdesc: !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>> -> vector<64xf32>
    gpu.return %data : vector<64xf32>
  }

  //-----

  // CHECK-LABEL: store_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.store_nd {{.*}}  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @store_nd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    xegpu.store_nd %data, %tdesc: vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: store_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-4: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.store_nd {{.*}}  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  gpu.func @store_nd_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = arith.constant dense<9.0> : vector<64xf32>
    xegpu.store_nd %data, %tdesc: vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: createNd_loadNd_storeNd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  //CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-COUNT-6: [[data:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  //CHECK: [[add:%.+]] = arith.addf {{.*}} : vector<24x32xf32>
  //CHECK-COUNT-6: [[extract:%.+]] = vector.extract_strided_slice {{.*}} : vector<24x32xf32> to vector<8x16xf32>
  //CHECK-COUNT-6: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @createNd_loadNd_storeNd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    %ld = xegpu.load_nd %tdesc: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    %add = arith.addf %data, %ld : vector<24x32xf32>
    xegpu.store_nd %add, %tdesc: vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
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

//-----

  // CHECK-LABEL: create_tdesc_vec
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  gpu.func @create_tdesc_vec(%src: ui64) -> !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>> {
    %cst = arith.constant dense<[
    0,   8,  16,  24,  32,  40,  48,  56,
    64,  72,  80,  88,  96, 104, 112, 120,
    128, 136, 144, 152, 160, 168, 176, 184,
    192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>,  #xegpu.layout<inst_data = [16]>>
  }

//-----

  // CHECK-LABEL: create_tdesc_step
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  gpu.func @create_tdesc_step(%src: ui64) -> !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>> {
    %step = arith.constant dense<8> : vector<32xindex>
    %seq = vector.step  : vector<32xindex>
    %cst = arith.muli %seq, %step : vector<32xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
  }

//-----

  // CHECK-LABEL: load
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.load  {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  gpu.func @load(%src: ui64) -> vector<32xf32> {
    %cst = arith.constant dense<[
    0,   8,  16,  24,  32,  40,  48,  56,
    64,  72,  80,  88,  96, 104, 112, 120,
    128, 136, 144, 152, 160, 168, 176, 184,
    192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    %ld = xegpu.load %tdesc, %mask: !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>, vector<32xi1> -> vector<32xf32>

    gpu.return %ld : vector<32xf32>
  }

//-----


  // CHECK-LABEL: load_with_offsets
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.load  {{.*}}[{{.*}}], {{.*}} <{chunk_size = 1 : i64, l1_hint = #xegpu.cache_hint<cached>}> : ui64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
  gpu.func @load_with_offsets(%src: ui64) -> vector<32xf32> {
      %cst = arith.constant dense<[
      0,   8,  16,  24,  32,  40,  48,  56,
      64,  72,  80,  88,  96, 104, 112, 120,
      128, 136, 144, 152, 160, 168, 176, 184,
      192, 200, 208, 216, 224, 232, 240, 248
      ]> : vector<32xindex>

      %c17 = arith.constant 17: index
      %mask = vector.create_mask %c17: vector<32xi1>
      %ld = xegpu.load %src[%cst], %mask {chunk_size = 1, layout_result_0 = #xegpu.layout<inst_data = [16]>, l1_hint = #xegpu.cache_hint<cached>} : ui64, vector<32xindex>, vector<32xi1> -> vector<32xf32>

      gpu.return %ld : vector<32xf32>
  }

//-----

  // CHECK-LABEL: prefetch
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.prefetch {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  gpu.func @prefetch(%src: ui64)  {

    %cst = arith.constant dense<[
    0,   8,  16,  24,  32,  40,  48,  56,
    64,  72,  80,  88,  96, 104, 112, 120,
    128, 136, 144, 152, 160, 168, 176, 184,
    192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>

    xegpu.prefetch %tdesc: !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

//-----

  // CHECK-LABEL: store
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.store  {{.*}} : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
  gpu.func @store(%src: ui64) {
    %cst = arith.constant dense<[
    0,   8,  16,  24,  32,  40,  48,  56,
    64,  72,  80,  88,  96, 104, 112, 120,
    128, 136, 144, 152, 160, 168, 176, 184,
    192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>

    %st_vec = arith.constant dense<1023.0>: vector<32xf32>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    xegpu.store %st_vec, %tdesc, %mask: vector<32xf32>, !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>, vector<32xi1>

    gpu.return
  }
  
  //-----

  // CHECK-LABEL: store_with_offsets
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.store  {{.*}}[{{.*}}], {{.*}} <{chunk_size = 1 : i64, l1_hint = #xegpu.cache_hint<cached>}> : vector<16xf32>, ui64, vector<16xindex>, vector<16xi1>
  gpu.func @store_with_offsets(%src: ui64) {
      %cst = arith.constant dense<[
      0,   8,  16,  24,  32,  40,  48,  56,
      64,  72,  80,  88,  96, 104, 112, 120,
      128, 136, 144, 152, 160, 168, 176, 184,
      192, 200, 208, 216, 224, 232, 240, 248
      ]> : vector<32xindex>

      %c17 = arith.constant 17: index
      %mask = vector.create_mask %c17: vector<32xi1>

      %st_vec = arith.constant dense<1023.0>: vector<32xf32>
      xegpu.store %st_vec, %src[%cst], %mask {chunk_size = 1, layout = #xegpu.layout<inst_data = [16]>, l1_hint = #xegpu.cache_hint<cached>} : vector<32xf32>, ui64, vector<32xindex>, vector<32xi1>

      gpu.return
  }

//-----
  // CHECK-LABEL: create_tdesc_step_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<chunk_size = 4 : i64>>
  gpu.func @create_tdesc_step_chunk(%src: ui64) -> !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 4]>> {
    %step = arith.constant dense<8> : vector<32xindex>
    %seq = vector.step  : vector<32xindex>
    %cst = arith.muli %seq, %step : vector<32xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 4]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 4]>>
  }

//-----
  // CHECK-LABEL: create_tdesc_step_chunk2
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-4: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  gpu.func @create_tdesc_step_chunk2(%src: ui64) -> !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>> {
    %step = arith.constant dense<8> : vector<32xindex>
    %seq = vector.step  : vector<32xindex>
    %cst = arith.muli %seq, %step : vector<32xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>
  }

// CHECK-LABEL: create_tdesc_step_chunk3
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<16xindex>
  // CHECK: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<16xindex>
  // CHECK: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
 // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<16xindex>
  // CHECK: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
    gpu.func @create_tdesc_step_chunk3(%src: ui64) -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size=8>, #xegpu.layout<inst_data = [16, 2]>> {
    %step = arith.constant dense<8> : vector<16xindex>
    %seq = vector.step  : vector<16xindex>
    %cst = arith.muli %seq, %step : vector<16xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32,  #xegpu.scatter_tdesc_attr<chunk_size=8>, #xegpu.layout<inst_data = [16, 2]>>
    gpu.return %tdesc : !xegpu.tensor_desc<16x8xf32,  #xegpu.scatter_tdesc_attr<chunk_size=8>, #xegpu.layout<inst_data = [16, 2]>>
  }

//-----
  // CHECK-LABEL: load_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-4: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-4: xegpu.load  {{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1> -> vector<16x2xf32>

  gpu.func @load_chunk(%src: ui64) -> vector<32x4xf32> {
    %cst = arith.constant dense<[
        0,   8,  16,  24,  32,  40,  48,  56,
        64,  72,  80,  88,  96, 104, 112, 120,
        128, 136, 144, 152, 160, 168, 176, 184,
        192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>

    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>
    %ld = xegpu.load %tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>, vector<32xi1> -> vector<32x4xf32>

    gpu.return %ld : vector<32x4xf32>
   }

//-----
  // CHECK-LABEL: load_with_offsets_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK: [[cst:%.+]] = arith.constant dense<0.000000e+00> : vector<32x4xf32>
  // CHECK: [[cst0:%.+]] = arith.constant dense<[130, 138, 146, 154, 162, 170, 178, 186, 194, 202, 210, 218, 226, 234, 242, 250]> : vector<16xindex>
  // CHECK: [[cst1:%.+]] = arith.constant dense<[2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122]> : vector<16xindex>
  // CHECK: [[cst2:%.+]] = arith.constant dense<[128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248]> : vector<16xindex>
  // CHECK: [[cst3:%.+]] = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
  // CHECK-COUNT-4: xegpu.load  {{.*}}[{{.*}}], {{.*}} <{chunk_size = 2 : i64, l1_hint = #xegpu.cache_hint<cached>}> : ui64, vector<16xindex>, vector<16xi1> -> vector<16x2xf32>
   gpu.func @load_with_offsets_chunk(%src: ui64) -> vector<32x4xf32> {
    %cst = arith.constant dense<[
        0,   8,  16,  24,  32,  40,  48,  56,
        64,  72,  80,  88,  96, 104, 112, 120,
        128, 136, 144, 152, 160, 168, 176, 184,
        192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>
    %ld = xegpu.load %src[%cst], %mask {chunk_size = 4, layout_result_0 = #xegpu.layout<inst_data = [16, 2]>, l1_hint = #xegpu.cache_hint<cached>} : ui64, vector<32xindex>, vector<32xi1> -> vector<32x4xf32>
    gpu.return %ld : vector<32x4xf32>
   }

//-----
  // CHECK-LABEL: store_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-4: xegpu.create_tdesc [[arg0]], {{.*}} :  ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-4: xegpu.store  {{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<16x2xf32>, !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1>
  gpu.func @store_chunk(%src: ui64) {
    %cst = arith.constant dense<[
      0,   8,  16,  24,  32,  40,  48,  56,
      64,  72,  80,  88,  96, 104, 112, 120,
      128, 136, 144, 152, 160, 168, 176, 184,
      192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>

    %st_vec = arith.constant dense<1023.>: vector<32x4xf32>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>
    xegpu.store %st_vec, %tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<32x4xf32>, !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16,2]>>, vector<32xi1>

    gpu.return
  }

//-----
  // CHECK-LABEL: store_with_offsets_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK: [[cst:%.+]] = arith.constant dense<1.023000e+03> : vector<16x2xf32
  // CHECK: [[cst0:%.+]] = arith.constant dense<[130, 138, 146, 154, 162, 170, 178, 186, 194, 202, 210, 218, 226, 234, 242, 250]> : vector<16xindex>
  // CHECK: [[cst1:%.+]] = arith.constant dense<[2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122]> : vector<16xindex>
  // CHECK: [[cst2:%.+]] = arith.constant dense<[128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248]> : vector<16xindex>
  // CHECK: [[cst3:%.+]] = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
  // CHECK-COUNT-4: xegpu.store  {{.*}}[{{.*}}], {{.*}} <{chunk_size = 2 : i64, l1_hint = #xegpu.cache_hint<cached>}> : vector<16x2xf32>, ui64, vector<16xindex>, vector<16xi1>
  gpu.func @store_with_offsets_chunk(%src: ui64) {
    %cst = arith.constant dense<[
      0,   8,  16,  24,  32,  40,  48,  56,
      64,  72,  80,  88,  96, 104, 112, 120,
      128, 136, 144, 152, 160, 168, 176, 184,
      192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>

    %st_vec = arith.constant dense<1023.>: vector<32x4xf32>
    xegpu.store %st_vec, %src[%cst], %mask {chunk_size = 4, layout = #xegpu.layout<inst_data = [16, 2]>, l1_hint = #xegpu.cache_hint<cached>} : vector<32x4xf32>, ui64, vector<32xindex>, vector<32xi1>
    gpu.return
  }

//-----
  // CHECK-LABEL: prefetch_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-2: xegpu.prefetch {{.*}} : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  gpu.func @prefetch_chunk(%src: ui64)  {
    %cst = arith.constant dense<[
      0,   8,  16,  24,  32,  40,  48,  56,
      64,  72,  80,  88,  96, 104, 112, 120,
      128, 136, 144, 152, 160, 168, 176, 184,
      192, 200, 208, 216, 224, 232, 240, 248
      ]> : vector<32xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>
    xegpu.prefetch %tdesc: !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>

    gpu.return
  }

//-----
  // CHECK-LABEL: update_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-4: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-4: xegpu.update_offset {{.*}} : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xindex>
  gpu.func @update_chunk(%src: ui64) -> !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>> {
    %cst = arith.constant dense<[
      0,   8,  16,  24,  32,  40,  48,  56,
      64,  72,  80,  88,  96, 104, 112, 120,
      128, 136, 144, 152, 160, 168, 176, 184,
      192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>
    %delta = arith.constant dense<32>: vector<32xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>

    %new_tdesc = xegpu.update_offset %tdesc, %delta
        : !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>, vector<32xindex>

    gpu.return %new_tdesc : !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>
  }
}

