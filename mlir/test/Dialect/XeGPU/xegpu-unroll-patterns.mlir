// RUN: mlir-opt --test-xegpu-unrolling-patterns -split-input-file %s | FileCheck %s

gpu.module @test {

  // CHECK-LABEL: test_create_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK: [[cast:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32>, !xegpu.tensor_desc<8x16xf32>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32>, !xegpu.tensor_desc<8x16xf32>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  // CHECK-SAME: to !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {__xegpu_blocking_tile_shape__ = array<i64: 8, 16>, __xegpu_blocking_unpack__}
  gpu.func @test_create_nd_tdesc(%src: memref<24x32xf32>) -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
  }

  //-----

  // CHECK-LABEL: test_create_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-2: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK: [[cast:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-SAME: !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>
  // CHECK-SAME: to !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {__xegpu_blocking_tile_shape__ = array<i64: 16>, __xegpu_blocking_unpack__}
  gpu.func @test_create_nd_tdesc_1d(%src: memref<64xf32>) -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
  }

  //-----

  // CHECK-LABEL: test_update_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[update:%.+]] = xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<8x16xf32>
  gpu.func @test_update_nd_tdesc(%src: memref<24x32xf32>) -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %update = xegpu.update_nd_offset %tdesc, [0, 16] : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return %update : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
  }

  //-----

  // CHECK-LABEL: test_update_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-2: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-2: [[update:%.+]] = xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<16xf32>
  gpu.func @test_update_nd_tdesc_1d(%src: memref<64xf32>) -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>> {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
    %update = xegpu.update_nd_offset %tdesc, [32] : !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return %update : !xegpu.tensor_desc<32xf32, #xegpu.layout<inst_data = [16]>>
  }

  //-----

  // CHECK-LABEL: test_prefetch_nd_tdesc
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.prefetch_nd {{.*}} : !xegpu.tensor_desc<8x16xf32>
  gpu.func @test_prefetch_nd_tdesc(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    xegpu.prefetch_nd %tdesc : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: test_prefetch_nd_tdesc_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-4: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.prefetch_nd {{.*}} : !xegpu.tensor_desc<16xf32>
  gpu.func @test_prefetch_nd_tdesc_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    xegpu.prefetch_nd %tdesc : !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

  //-----
  // CHECK-LABEL: test_load_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: [[ld:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  // CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  gpu.func @test_load_nd(%src: memref<24x32xf32>) -> vector<24x32xf32> {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %ld = xegpu.load_nd %tdesc: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    gpu.return %ld : vector<24x32xf32>
  }

  //-----

  // CHECK-LABEL: test_load_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-4: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: [[ld:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
  // CHECK-COUNT-4: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<16xf32> into vector<64xf32>
  gpu.func @test_load_nd_1d(%src: memref<64xf32>) -> vector<64xf32> {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = xegpu.load_nd %tdesc: !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>> -> vector<64xf32>
    gpu.return %data : vector<64xf32>
  }

  //-----

  // CHECK-LABEL: test_store_nd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  // CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK-COUNT-6: xegpu.store_nd {{.*}}  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @test_store_nd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    xegpu.store_nd %data, %tdesc: vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: test_store_nd_1d
  // CHECK-SAME: [[arg0:%.+]]: memref<64xf32>
  // CHECK-COUNT-4: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<64xf32> -> !xegpu.tensor_desc<16xf32>
  // CHECK-COUNT-4: xegpu.store_nd {{.*}}  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  gpu.func @test_store_nd_1d(%src: memref<64xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0] : memref<64xf32> -> !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    %data = arith.constant dense<9.0> : vector<64xf32>
    xegpu.store_nd %data, %tdesc: vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.layout<inst_data = [16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: test_createNd_loadNd_storeNd
  // CHECK-SAME: [[arg0:%.+]]: memref<24x32xf32>
  //CHECK-COUNT-6: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[arg0]][{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-COUNT-6: [[data:%.+]] = xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-COUNT-6: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<24x32xf32>
  //CHECK: [[add:%.+]] = arith.addf {{.*}} : vector<24x32xf32>
  //CHECK-COUNT-6: [[extract:%.+]] = vector.extract_strided_slice {{.*}} : vector<24x32xf32> to vector<8x16xf32>
  //CHECK-COUNT-6: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.func @test_createNd_loadNd_storeNd(%src: memref<24x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    %data = arith.constant dense<9.0> : vector<24x32xf32>
    %ld = xegpu.load_nd %tdesc: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<24x32xf32>
    %add = arith.addf %data, %ld : vector<24x32xf32>
    xegpu.store_nd %add, %tdesc: vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<inst_data = [8, 16]>>
    gpu.return
  }

  //-----

  // CHECK-LABEL: test_dpas
  // CHECK-SAME: [[arg0:%.+]]: vector<32x32xf16>, [[arg1:%.+]]: vector<32x32xf16>
  //CHECK-COUNT-8: [[extract1:%.+]] = vector.extract_strided_slice [[arg0]] {{.*}} : vector<32x32xf16> to vector<8x16xf16>
  //CHECK-COUNT-4: [[extract2:%.+]] = vector.extract_strided_slice [[arg1]] {{.*}} : vector<32x32xf16> to vector<16x16xf16>
  //CHECK-COUNT-16: [[dpas:%.+]] = xegpu.dpas {{.*}} -> vector<8x16xf32>
  //CHECK-COUNT-8: [[insert:%.+]] = vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<32x32xf32>
  gpu.func @test_dpas(%a: vector<32x32xf16>, %b: vector<32x32xf16>) -> vector<32x32xf32> {
    %c = xegpu.dpas %a, %b : vector<32x32xf16>, vector<32x32xf16> -> vector<32x32xf32>
    gpu.return %c : vector<32x32xf32>
  }

//-----

  // CHECK-LABEL: test_create_tdesc_vec
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  gpu.func @test_create_tdesc_vec(%src: ui64) -> !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>> {
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

  // CHECK-LABEL: test_create_tdesc_step
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  gpu.func @test_create_tdesc_step(%src: ui64) -> !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>> {
    %step = arith.constant dense<8> : vector<32xindex>
    %seq = vector.step  : vector<32xindex>
    %cst = arith.muli %seq, %step : vector<32xindex>
    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    gpu.return %tdesc : !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
  }

//-----

  // CHECK-LABEL: test_load
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.load  {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  gpu.func @test_load(%src: ui64) -> vector<32xf32> {
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

  // CHECK-LABEL: test_prefetch
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.prefetch {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  gpu.func @test_prefetch(%src: ui64)  {

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

  // CHECK-LABEL: test_store
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.store  {{.*}} : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
  gpu.func @test_store(%src: ui64) {
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

  // CHECK-LABEL: test_prefetch_load_store_update
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.prefetch {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
   // CHECK-COUNT-2: xegpu.update_offset {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
   // CHECK-COUNT-2: xegpu.load  {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  // CHECK-COUNT-2: xegpu.store  {{.*}} : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>

  gpu.func @test_prefetch_load_store_update(%src: ui64)  {

    %cst = arith.constant dense<[
    0,   8,  16,  24,  32,  40,  48,  56,
    64,  72,  80,  88,  96, 104, 112, 120,
    128, 136, 144, 152, 160, 168, 176, 184,
    192, 200, 208, 216, 224, 232, 240, 248 
    ]> : vector<32xindex>

    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    xegpu.prefetch %tdesc: !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
   
    %delta = arith.constant dense<[
    32,   32,  32,  32,  32,  32,  32,  32,
    32,   32,  32,  32,  32,  32,  32,  64,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 256 
    ]> : vector<32xindex>
    %new_tdesc = xegpu.update_offset %tdesc, %delta
              : !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>, vector<32xindex>     
 
    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>

    %ld_vec = xegpu.load %new_tdesc, %mask: !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>, vector<32xi1> -> vector<32xf32>

    %st_vec = arith.addf %ld_vec, %ld_vec : vector<32xf32>
    xegpu.store %st_vec, %tdesc, %mask: 
                 vector<32xf32>, 
                 !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>, 
                 vector<32xi1>
  
    gpu.return
  }
}
