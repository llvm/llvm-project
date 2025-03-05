func.func @test(%a: memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) {
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
func.func @test(%a: memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) {
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
func.func @test(%a: memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) {
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
func.func @test(%arg0: vector<8x32xi8>, %arg1: vector<32x16xi8>, %c : memref<8x16xi32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.dpas %arg0, %arg1 : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
  %1 = xegpu.create_nd_tdesc %c [%c0, %c0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
  xegpu.store_nd %0, %1 : vector<8x16xi32>, !xegpu.tensor_desc<8x16xi32>
  return
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.extf %1 : vector<16x16xf16> to vector<16x16xf32>
  %3 = arith.truncf %2 : vector<16x16xf32> to vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----
func.func @test_load_gather(%a: memref<8x16xf16>, %b : memref<256xf16>, %c : memref<8x16xf32>) {
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
func.func @test_bitcast(%arg0: memref<8x16xi16>, %arg1: memref<32x16xi8>, %c : memref<8x16xi32>) {
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
func.func @test_bitcast(%arg0: memref<8x32xi8>, %arg1: memref<16x32xi8>, %c : memref<8x16xf32>) {
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
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %cst = arith.constant dense<1.000000e+00> : vector<16x16xf16>
  %2 = arith.addf %1, %cst : vector<16x16xf16>
  %3 = xegpu.dpas %0, %2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4, %2 : vector<8x16xf32>, vector<16x16xf16>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %3:2 = scf.for %arg3 = %c0 to %arg2 step %c1 iter_args(%arg4 = %1, %arg5 = %2) -> (vector<16x16xf16>, vector<8x16xf32>) {
    %4 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %5 = arith.addf %arg4, %4 : vector<16x16xf16>
    %6 = xegpu.dpas %0, %5 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    scf.yield %5, %6 : vector<16x16xf16>, vector<8x16xf32>
  }
  return %3#1 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %1 : vector<8x16xf32>, vector<16x16xf16>
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
func.func @test(%arg0: !xegpu.tensor_desc<8x16xi16>, %arg1: !xegpu.tensor_desc<16x16xi16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xi16> -> vector<8x16xi16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xi16> -> vector<16x16xi16>
  %2 = arith.bitcast %0 : vector<8x16xi16> to vector<8x16xf16>
  %3 = arith.bitcast %1 : vector<16x16xi16> to vector<16x16xf16>
  %4 = xegpu.dpas %2, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
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
