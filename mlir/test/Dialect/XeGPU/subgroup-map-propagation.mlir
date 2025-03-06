func.func @test_dpas_op_1(%a: memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.0> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %a[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %b[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %c[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_dpas_op_2(%arg0: vector<8x32xi8>, %arg1: vector<32x16xi8>, %c : memref<8x16xi32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.dpas %arg0, %arg1 : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
  %1 = xegpu.create_nd_tdesc %c [%c0, %c0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
  xegpu.store_nd %0, %1 : vector<8x16xi32>, !xegpu.tensor_desc<8x16xi32>
  return
}

// -----
func.func @test_transpose_op_1(%a: memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.0> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %a[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %b[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 {transpose = array<i64: 1, 0>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %c[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_transpose_op_2(%a: memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.0> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %a[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %b[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %6 = vector.transpose %3, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %4 = xegpu.dpas %2, %6, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %c[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}



// -----
func.func @test_extf_truncf_op(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.extf %1 : vector<16x16xf16> to vector<16x16xf32>
  %3 = arith.truncf %2 : vector<16x16xf32> to vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----
func.func @test_load_gather_op(%a: memref<8x16xf16>, %b : memref<256xf16>, %c : memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %6 = xegpu.create_nd_tdesc %a[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %7 = xegpu.load_nd %6 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %0 = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %1 = arith.constant dense<1>: vector<16xi1>
  %2 = xegpu.create_tdesc %b, %0 : memref<256xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16>>
  %3 = xegpu.load %2, %1 <{transpose}> : !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16>>, vector<16xi1> -> vector<16x16xf16>
  %4 = xegpu.dpas %7, %3 : vector<8x16xf16>, vector<16x16xf16>-> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %c[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_store_scatter_op(%c: memref<128xf32>){
  %cst = arith.constant dense<1.000000e+00> : vector<8x16xf32>
  %1 = arith.constant dense<1> : vector<16xi1>
  %2 = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %3 = xegpu.create_tdesc %c, %2 : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  xegpu.store %cst, %3, %1 <{transpose}> : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1>
  return
}

// -----
func.func @test_bitcast_op_1(%arg0: memref<8x16xi16>, %arg1: memref<32x16xi8>, %c : memref<8x16xi32>) {
  %c0 = arith.constant 0 : index
  %2 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xi16> -> !xegpu.tensor_desc<8x16xi16>
  %3 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<32x16xi8> -> !xegpu.tensor_desc<32x16xi8>
  %4 = xegpu.load_nd %2 : !xegpu.tensor_desc<8x16xi16> -> vector<8x16xi16>
  %5 = xegpu.load_nd %3 : !xegpu.tensor_desc<32x16xi8> -> vector<32x16xi8>
  %6 = vector.bitcast %4 : vector<8x16xi16> to vector<8x32xi8>
  %0 = xegpu.dpas %6, %5 : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
  %1 = xegpu.create_nd_tdesc %c [%c0, %c0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
  xegpu.store_nd %0, %1 : vector<8x16xi32>, !xegpu.tensor_desc<8x16xi32>
  return
}

// -----
func.func @test_bitcast_op_2(%arg0: memref<8x32xi8>, %arg1: memref<16x32xi8>, %c : memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %2 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x32xi8> -> !xegpu.tensor_desc<8x32xi8>
  %3 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x32xi8> -> !xegpu.tensor_desc<16x32xi8>
  %4 = xegpu.load_nd %2 : !xegpu.tensor_desc<8x32xi8> -> vector<8x32xi8>
  %5 = xegpu.load_nd %3 : !xegpu.tensor_desc<16x32xi8> -> vector<16x32xi8>
  %6 = vector.bitcast %4 : vector<8x32xi8> to vector<8x16xf16>
  %7 = vector.bitcast %5 : vector<16x32xi8> to vector<16x16xf16>
  %0 = xegpu.dpas %6, %7 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %1 = xegpu.create_nd_tdesc %c [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %0, %1 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_binary_op_1(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: !xegpu.tensor_desc<8x16xf32>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %4, %arg2 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_binary_op_2(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: !xegpu.tensor_desc<8x16xf32>, %arg3: !xegpu.tensor_desc<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %cst = arith.constant dense<1.000000e+00> : vector<16x16xf16>
  %2 = arith.addf %1, %cst : vector<16x16xf16>
  %3 = xegpu.dpas %0, %2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %3, %arg2 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %2, %arg3 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  return
}

// -----
func.func @test_for_op(%a: memref<8x128xf16>, %b : memref<128x16xf16>, %c : memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %0 = xegpu.create_nd_tdesc %a[%c0, %c0] : memref<8x128xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %b[%c0, %c0] : memref<128x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = arith.constant dense<0.0> : vector<8x16xf32>
  %3:3 = scf.for %k = %c0 to %c128 step %c16 iter_args(%arg0 = %0, %arg1 = %1, %arg2 = %2) -> (!xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>) {
    %4 = xegpu.load_nd %arg0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %5 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %6 = xegpu.dpas %4, %5, %arg2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %7 = xegpu.update_nd_offset %arg0, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
    %8 = xegpu.update_nd_offset %arg1, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
    scf.yield %7, %8, %6 : !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>
  }
  %9 = xegpu.create_nd_tdesc %c[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %3#2, %9 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_if_op_1(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1, %arg3: !xegpu.tensor_desc<8x16xf32>){
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %2, %arg3 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_if_op_2(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1, %arg3: !xegpu.tensor_desc<8x16xf32>, %arg4: !xegpu.tensor_desc<16x16xf16>){
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %2, %arg3 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %1, %arg4 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  return
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: vector<16x16xf16>, %arg3: i1) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg3 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    scf.yield %arg2 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: vector<16x16xf16>, %arg3: i1) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg3 -> (vector<8x16xf32>) {
    %2 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %3 = arith.addf %2, %arg2 : vector<16x16xf16>
    %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    scf.yield %4 : vector<8x16xf32>
  } else {
    %2 = xegpu.dpas %0, %arg2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    scf.yield %2 : vector<8x16xf32>
  }
  return %1 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xi16>, %arg1: !xegpu.tensor_desc<16x16xi16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xi16> -> vector<8x16xi16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xi16> -> vector<16x16xi16>
  %2 = arith.bitcast %0 : vector<8x16xi16> to vector<8x16xf16>
  %3 = arith.bitcast %1 : vector<16x16xi16> to vector<16x16xf16>
  %4 = xegpu.dpas %2, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}
