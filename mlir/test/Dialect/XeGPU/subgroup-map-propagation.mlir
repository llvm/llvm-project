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
func.func @test(%arg0: vector<8x16xf16>, %arg1: vector<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.dpas %arg0, %arg1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: vector<8x32xi8>, %arg1: vector<32x32xi8>) -> vector<8x32xi32> {
  %0 = xegpu.dpas %arg0, %arg1 : vector<8x32xi8>, vector<32x32xi8> -> vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<8x16xf32>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %3 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %3 : vector<8x16xf32>, vector<8x16xf32>
}

// -----
func.func @test(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %1 : vector<8x16xf32>, vector<16x16xf16>
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
