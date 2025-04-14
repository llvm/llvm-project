gpu.module @test {
gpu.func @test_store_nd_1d(%arg0: memref<16xf32>){
  %c0 = arith.constant 0 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
  xegpu.store_nd %1, %0 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  gpu.return
}
}

// // -----
// gpu.module @test {
// gpu.func @test_store_nd_2d(%arg0: memref<16x16xf16>){
//   %c0 = arith.constant 0 : index
//   %1 = arith.constant dense<1.000000e+00> : vector<16x16xf16>
//   %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
//   xegpu.store_nd %1, %0 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
//   gpu.return
// }
// }



// // -----
// gpu.module @test {
// gpu.func @test_load_nd_1d(%arg0: memref<16xf32>, %arg1: memref<16xf32>){
//   %c0 = arith.constant 0 : index
//   %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
//   %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
//   %2 = xegpu.create_nd_tdesc %arg1[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
//   xegpu.store_nd %1, %2 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
//   gpu.return
// }
// }

// // -----
// gpu.module @test {
// gpu.func @test_load_nd_2d(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>){
//   %c0 = arith.constant 0 : index
//   %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
//   %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//   %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
//   xegpu.store_nd %1, %2 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
//   gpu.return
// }
// }

// // -----
// gpu.module @test {
// gpu.func @test_load_nd_array_length(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>){
//   %c0 = arith.constant 0 : index
//   %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
//   %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
//   %2 = vector.extract %1[%c0] : vector<16x16xf16> from vector<2x16x16xf16>
//   %3 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
//   xegpu.store_nd %2, %3 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
//   gpu.return
// }
// }

// // -----
// gpu.module @test {
// gpu.func @test_dpas(%arg0: vector<8x16xf16>, %arg1: vector<16x16xf16>, %arg3: vector<8x16xf32>, %arg2: memref<8x16xf32>){
//   %c0 = arith.constant 0 : index
//   %0 = xegpu.dpas %arg0, %arg1, %arg3 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
//   %3 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
//   xegpu.store_nd %0, %3 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
//   gpu.return
// }
// }
