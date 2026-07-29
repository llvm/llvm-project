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
  // CHECK-COUNT-6: xegpu.prefetch_nd {{.*}}[{{.*}}] : !xegpu.tensor_desc<8x16xf32>
  gpu.func @prefetch_nd_tdesc(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    xegpu.prefetch_nd %tdesc[0, 0] : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

//-----
  // CHECK-LABEL: prefetch_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.prefetch_nd {{.*}}[{{.*}}] : !xegpu.tensor_desc<16xf32>
  gpu.func @prefetch_nd_tdesc_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    xegpu.prefetch_nd %tdesc[0] : !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

//-----
  // CHECK-LABEL: load_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[ld:%.+]] = xegpu.load_nd {{.*}}[{{.*}}]  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  // CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  gpu.func @load_nd(%src: memref<24x32xf32>) -> vector<24x32xf32> {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %ld = xegpu.load_nd %tdesc[0, 0] : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    gpu.return %ld : vector<24x32xf32>
  }

//-----
  // CHECK-LABEL: load_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: [[ld:%.+]] = xegpu.load_nd {{.*}}[{{.*}}]  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
  // CHECK-COUNT-4: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<16xf32> into vector<64xf32>
  gpu.func @load_nd_1d(%src: memref<64xf32>) -> vector<64xf32> {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = xegpu.load_nd %tdesc[0] : !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>> -> vector<64xf32>
    gpu.return %data : vector<64xf32>
  }

//-----
  // CHECK-LABEL: store_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.store_nd {{.*}}[{{.*}}]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @store_nd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    xegpu.store_nd %data, %tdesc[0, 0] : vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

//-----
  // CHECK-LABEL: store_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.store_nd {{.*}}[{{.*}}]  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  gpu.func @store_nd_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = arith.constant dense<9.0> : vector<64xf32>
    xegpu.store_nd %data, %tdesc[0] : vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

//-----
  // CHECK-LABEL: createNd_loadNd_storeNd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  //CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-COUNT-6: [[data:%.+]] = xegpu.load_nd {{.*}}[{{.*}}]  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  //CHECK: [[add:%.+]] = arith.addf {{.*}} : vector<24x32xf32>
  //CHECK-COUNT-6: [[extract:%.+]] = vector.extract_strided_slice {{.*}} : vector<24x32xf32> to vector<8x16xf32>
  //CHECK-COUNT-6: xegpu.store_nd {{.*}}[{{.*}}] : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @createNd_loadNd_storeNd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    %ld = xegpu.load_nd %tdesc[0, 0] : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    %add = arith.addf %data, %ld : vector<24x32xf32>
    xegpu.store_nd %add, %tdesc[0, 0] : vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
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
      %ld = xegpu.load %src[%cst], %mask {chunk_size = 1, layout = #xegpu.layout<inst_data = [16]>, l1_hint = #xegpu.cache_hint<cached>} : ui64, vector<32xindex>, vector<32xi1> -> vector<32xf32>

      gpu.return %ld : vector<32xf32>
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
    %ld = xegpu.load %src[%cst], %mask {chunk_size = 4, layout = #xegpu.layout<inst_data = [16, 2]>, l1_hint = #xegpu.cache_hint<cached>} : ui64, vector<32xindex>, vector<32xi1> -> vector<32x4xf32>
    gpu.return %ld : vector<32x4xf32>
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
  // CHECK-LABEL: multi_reduction_2d_last_dim
  // CHECK-SAME: [[SRC:%.+]]: vector<32x80xf32>, [[ACC:%.+]]: vector<32xf32>
  //
  // Extract column tiles for the first row-tile:
  // CHECK: [[TILE00:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 0]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE01:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 16]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE02:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 32]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE03:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 48]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE04:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 64]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  //
  // Perform sequential reduction for first tile (rows 0..15):
  // CHECK: [[TMP00:%.+]] = arith.addf [[TILE00]], [[TILE01]] : vector<16x16xf32>
  // CHECK: [[TMP01:%.+]] = arith.addf [[TMP00]], [[TILE02]] : vector<16x16xf32>
  // CHECK: [[TMP02:%.+]] = arith.addf [[TMP01]], [[TILE03]] : vector<16x16xf32>
  // CHECK: [[TMP03:%.+]] = arith.addf [[TMP02]], [[TILE04]] : vector<16x16xf32>
  // CHECK: [[ACC0:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [0]{{.*}} : vector<32xf32> to vector<16xf32>
  // CHECK: [[RED0:%.+]] = vector.multi_reduction <add>, [[TMP03]], [[ACC0]] [1] : vector<16x16xf32> to vector<16xf32>
  // CHECK: [[INS0:%.+]] = vector.insert_strided_slice [[RED0]], {{%.+}} {offsets = [0]{{.*}} : vector<16xf32> into vector<32xf32>
  //
  // Extract column tiles for the second row-tile:
  // CHECK: [[TILE10:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 0]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE11:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 16]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE12:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 32]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE13:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 48]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  // CHECK: [[TILE14:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 64]{{.*}} : vector<32x80xf32> to vector<16x16xf32>
  //
  // Perform sequential reduction for second tile (rows 16..31):
  // CHECK: [[TMP10:%.+]] = arith.addf [[TILE10]], [[TILE11]] : vector<16x16xf32>
  // CHECK: [[TMP11:%.+]] = arith.addf [[TMP10]], [[TILE12]] : vector<16x16xf32>
  // CHECK: [[TMP12:%.+]] = arith.addf [[TMP11]], [[TILE13]] : vector<16x16xf32>
  // CHECK: [[TMP13:%.+]] = arith.addf [[TMP12]], [[TILE14]] : vector<16x16xf32>
  // CHECK: [[ACC1:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [16]{{.*}} : vector<32xf32> to vector<16xf32>
  // CHECK: [[RED1:%.+]] = vector.multi_reduction <add>, [[TMP13]], [[ACC1]] [1] : vector<16x16xf32> to vector<16xf32>
  // CHECK: [[INS1:%.+]] = vector.insert_strided_slice [[RED1]], [[INS0]] {offsets = [16]{{.*}} : vector<16xf32> into vector<32xf32>
  gpu.func @multi_reduction_2d_last_dim(%src: vector<32x80xf32>, %acc: vector<32xf32>) -> vector<32xf32> {
    %0 = vector.multi_reduction <add>, %src, %acc {layout_operand_0 = #xegpu.layout<inst_data = [16, 16]>} [1] : vector<32x80xf32> to vector<32xf32>
    gpu.return %0 : vector<32xf32>
  }

//-----
  // Reduction over multiple dimensions [1, 3] in a 4D vector.
  // source: <4x8x16x32xf32>, target shape: <2x4x16x16xf32>
  //
  // CHECK-LABEL: multi_reduction_multi_dim
  // CHECK-SAME: [[SRC:%.+]]: vector<4x8x16x32xf32>, [[ACC:%.+]]: vector<4x16xf32>
  //
  // First kept tile [0, 0]: extract 4 source tiles over reduced dims [1, 3]
  // CHECK: [[T00:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 0, 0, 0], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // CHECK: [[T01:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 0, 0, 16], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // CHECK: [[T02:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 4, 0, 0], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // CHECK: [[T03:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 4, 0, 16], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // Sequential reduction:
  // CHECK: [[R00:%.+]] = arith.addf [[T00]], [[T01]] : vector<2x4x16x16xf32>
  // CHECK: [[R01:%.+]] = arith.addf [[R00]], [[T02]] : vector<2x4x16x16xf32>
  // CHECK: [[R02:%.+]] = arith.addf [[R01]], [[T03]] : vector<2x4x16x16xf32>
  // CHECK: [[ACC0:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [0, 0], sizes = [2, 16]{{.*}} : vector<4x16xf32> to vector<2x16xf32>
  // CHECK: [[MR0:%.+]] = vector.multi_reduction <add>, [[R02]], [[ACC0]] [1, 3] : vector<2x4x16x16xf32> to vector<2x16xf32>
  // CHECK: [[INS0:%.+]] = vector.insert_strided_slice [[MR0]], {{%.+}} {offsets = [0, 0]{{.*}} : vector<2x16xf32> into vector<4x16xf32>
  //
  // Second kept tile [2, 0]:
  // CHECK: [[T10:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [2, 0, 0, 0], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // CHECK: [[T11:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [2, 0, 0, 16], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // CHECK: [[T12:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [2, 4, 0, 0], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // CHECK: [[T13:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [2, 4, 0, 16], sizes = [2, 4, 16, 16]{{.*}} : vector<4x8x16x32xf32> to vector<2x4x16x16xf32>
  // Sequential reduction:
  // CHECK: [[R10:%.+]] = arith.addf [[T10]], [[T11]] : vector<2x4x16x16xf32>
  // CHECK: [[R11:%.+]] = arith.addf [[R10]], [[T12]] : vector<2x4x16x16xf32>
  // CHECK: [[R12:%.+]] = arith.addf [[R11]], [[T13]] : vector<2x4x16x16xf32>
  // CHECK: [[ACC1:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [2, 0], sizes = [2, 16]{{.*}} : vector<4x16xf32> to vector<2x16xf32>
  // CHECK: [[MR1:%.+]] = vector.multi_reduction <add>, [[R12]], [[ACC1]] [1, 3] : vector<2x4x16x16xf32> to vector<2x16xf32>
  // CHECK: [[INS1:%.+]] = vector.insert_strided_slice [[MR1]], [[INS0]] {offsets = [2, 0]{{.*}} : vector<2x16xf32> into vector<4x16xf32>
  gpu.func @multi_reduction_multi_dim(%src: vector<4x8x16x32xf32>, %acc: vector<4x16xf32>) -> vector<4x16xf32> {
    %0 = vector.multi_reduction <add>, %src, %acc {layout_operand_0 = #xegpu.layout<inst_data = [2, 4, 16, 16]>} [1, 3] : vector<4x8x16x32xf32> to vector<4x16xf32>
    gpu.return %0 : vector<4x16xf32>
  }

//-----
  // Reduction over dimension [0] in a 2D vector.
  // source: <48x32xf32>, target shape: <16x16xf32>
  //
  // CHECK-LABEL: multi_reduction_reduce_dim0
  // CHECK-SAME: [[SRC:%.+]]: vector<48x32xf32>, [[ACC:%.+]]: vector<32xf32>
  //
  // First column tile [0]: extract 3 source tiles over reduced dim [0]
  // CHECK: [[T00:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 0], sizes = [16, 16]{{.*}} : vector<48x32xf32> to vector<16x16xf32>
  // CHECK: [[T01:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 0], sizes = [16, 16]{{.*}} : vector<48x32xf32> to vector<16x16xf32>
  // CHECK: [[T02:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [32, 0], sizes = [16, 16]{{.*}} : vector<48x32xf32> to vector<16x16xf32>
  // Sequential reduction:
  // CHECK: [[R00:%.+]] = arith.addf [[T00]], [[T01]] : vector<16x16xf32>
  // CHECK: [[R01:%.+]] = arith.addf [[R00]], [[T02]] : vector<16x16xf32>
  // CHECK: [[ACC0:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [0], sizes = [16]{{.*}} : vector<32xf32> to vector<16xf32>
  // CHECK: [[MR0:%.+]] = vector.multi_reduction <add>, [[R01]], [[ACC0]] [0] : vector<16x16xf32> to vector<16xf32>
  // CHECK: [[INS0:%.+]] = vector.insert_strided_slice [[MR0]], {{%.+}} {offsets = [0]{{.*}} : vector<16xf32> into vector<32xf32>
  //
  // Second column tile [16]:
  // CHECK: [[T10:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 16], sizes = [16, 16]{{.*}} : vector<48x32xf32> to vector<16x16xf32>
  // CHECK: [[T11:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 16], sizes = [16, 16]{{.*}} : vector<48x32xf32> to vector<16x16xf32>
  // CHECK: [[T12:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [32, 16], sizes = [16, 16]{{.*}} : vector<48x32xf32> to vector<16x16xf32>
  // Sequential reduction:
  // CHECK: [[R10:%.+]] = arith.addf [[T10]], [[T11]] : vector<16x16xf32>
  // CHECK: [[R11:%.+]] = arith.addf [[R10]], [[T12]] : vector<16x16xf32>
  // CHECK: [[ACC1:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [16], sizes = [16]{{.*}} : vector<32xf32> to vector<16xf32>
  // CHECK: [[MR1:%.+]] = vector.multi_reduction <add>, [[R11]], [[ACC1]] [0] : vector<16x16xf32> to vector<16xf32>
  // CHECK: [[INS1:%.+]] = vector.insert_strided_slice [[MR1]], [[INS0]] {offsets = [16]{{.*}} : vector<16xf32> into vector<32xf32>
  gpu.func @multi_reduction_reduce_dim0(%src: vector<48x32xf32>, %acc: vector<32xf32>) -> vector<32xf32> {
    %0 = vector.multi_reduction <add>, %src, %acc {layout_operand_0 = #xegpu.layout<inst_data = [16, 16]>} [0] : vector<48x32xf32> to vector<32xf32>
    gpu.return %0 : vector<32xf32>
  }

//-----
  // source: <32x16xf32>, target tile: <16x16xf32>
  // Verifies that the patterns works correctly when there is
  // no place for the sequential elementwise 'arith' reduction.
  //
  // CHECK-LABEL: multi_reduction_no_elwise
  // CHECK-SAME: [[SRC:%.+]]: vector<32x16xf32>, [[ACC:%.+]]: vector<32xf32>
  //
  // First row tile [0]: single source tile, no arith reduction
  // CHECK: [[T0:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [0, 0], sizes = [16, 16]{{.*}} : vector<32x16xf32> to vector<16x16xf32>
  // CHECK: [[ACC0:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [0], sizes = [16]{{.*}} : vector<32xf32> to vector<16xf32>
  // CHECK: [[MR0:%.+]] = vector.multi_reduction <add>, [[T0]], [[ACC0]] [1] : vector<16x16xf32> to vector<16xf32>
  // CHECK: [[INS0:%.+]] = vector.insert_strided_slice [[MR0]], {{%.+}} {offsets = [0]{{.*}} : vector<16xf32> into vector<32xf32>
  //
  // Second row tile [16]: single source tile, no arith reduction
  // CHECK: [[T1:%.+]] = vector.extract_strided_slice [[SRC]] {offsets = [16, 0], sizes = [16, 16]{{.*}} : vector<32x16xf32> to vector<16x16xf32>
  // CHECK: [[ACC1:%.+]] = vector.extract_strided_slice [[ACC]] {offsets = [16], sizes = [16]{{.*}} : vector<32xf32> to vector<16xf32>
  // CHECK: [[MR1:%.+]] = vector.multi_reduction <add>, [[T1]], [[ACC1]] [1] : vector<16x16xf32> to vector<16xf32>
  // CHECK: [[INS1:%.+]] = vector.insert_strided_slice [[MR1]], [[INS0]] {offsets = [16]{{.*}} : vector<16xf32> into vector<32xf32>
  gpu.func @multi_reduction_no_elwise(%src: vector<32x16xf32>, %acc: vector<32xf32>) -> vector<32xf32> {
    %0 = vector.multi_reduction <add>, %src, %acc {layout_operand_0 = #xegpu.layout<inst_data = [16, 16]>} [1] : vector<32x16xf32> to vector<32xf32>
    gpu.return %0 : vector<32xf32>
  }

}

