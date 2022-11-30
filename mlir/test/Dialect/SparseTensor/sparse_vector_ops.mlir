// RUN: mlir-opt %s -sparsification -cse -sparse-vectorization="vl=8" -cse | \
// RUN:   FileCheck %s

#DenseVector = #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) ops b(i)"
}

// CHECK-LABEL: func.func @vops
// CHECK-DAG:       %[[C1:.*]] = arith.constant dense<2.000000e+00> : vector<8xf32>
// CHECK-DAG:       %[[C2:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-DAG:       %[[C3:.*]] = arith.constant dense<255> : vector<8xi64>
// CHECK:           scf.for
// CHECK:             %[[VAL_14:.*]] = vector.load
// CHECK:             %[[VAL_15:.*]] = math.absf %[[VAL_14]] : vector<8xf32>
// CHECK:             %[[VAL_16:.*]] = math.ceil %[[VAL_15]] : vector<8xf32>
// CHECK:             %[[VAL_17:.*]] = math.floor %[[VAL_16]] : vector<8xf32>
// CHECK:             %[[VAL_18:.*]] = math.sqrt %[[VAL_17]] : vector<8xf32>
// CHECK:             %[[VAL_19:.*]] = math.expm1 %[[VAL_18]] : vector<8xf32>
// CHECK:             %[[VAL_20:.*]] = math.sin %[[VAL_19]] : vector<8xf32>
// CHECK:             %[[VAL_21:.*]] = math.tanh %[[VAL_20]] : vector<8xf32>
// CHECK:             %[[VAL_22:.*]] = arith.negf %[[VAL_21]] : vector<8xf32>
// CHECK:             %[[VAL_23:.*]] = vector.load
// CHECK:             %[[VAL_24:.*]] = arith.mulf %[[VAL_22]], %[[VAL_23]] : vector<8xf32>
// CHECK:             %[[VAL_25:.*]] = arith.divf %[[VAL_24]], %[[C1]] : vector<8xf32>
// CHECK:             %[[VAL_26:.*]] = arith.addf %[[VAL_25]], %[[C1]] : vector<8xf32>
// CHECK:             %[[VAL_27:.*]] = arith.subf %[[VAL_26]], %[[C2]] : vector<8xf32>
// CHECK:             %[[VAL_28:.*]] = arith.extf %[[VAL_27]] : vector<8xf32> to vector<8xf64>
// CHECK:             %[[VAL_29:.*]] = arith.bitcast %[[VAL_28]] : vector<8xf64> to vector<8xi64>
// CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_29]] : vector<8xi64>
// CHECK:             %[[VAL_31:.*]] = arith.andi %[[VAL_30]], %[[C3]] : vector<8xi64>
// CHECK:             %[[VAL_32:.*]] = arith.trunci %[[VAL_31]] : vector<8xi64> to vector<8xi16>
// CHECK:             %[[VAL_33:.*]] = arith.extsi %[[VAL_32]] : vector<8xi16> to vector<8xi32>
// CHECK:             %[[VAL_34:.*]] = arith.uitofp %[[VAL_33]] : vector<8xi32> to vector<8xf32>
// CHECK:             vector.store %[[VAL_34]]
// CHECK:           }
func.func @vops(%arga: tensor<1024xf32, #DenseVector>,
                %argb: tensor<1024xf32, #DenseVector>) -> tensor<1024xf32> {
  %init = bufferization.alloc_tensor() : tensor<1024xf32>
  %o = arith.constant 1.0 : f32
  %c = arith.constant 2.0 : f32
  %i = arith.constant 255 : i64
  %0 = linalg.generic #trait
    ins(%arga, %argb: tensor<1024xf32, #DenseVector>, tensor<1024xf32, #DenseVector>)
    outs(%init: tensor<1024xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = math.absf %a : f32
        %1 = math.ceil %0 : f32
        %2 = math.floor %1 : f32
        %3 = math.sqrt %2 : f32
        %4 = math.expm1 %3 : f32
        %5 = math.sin %4 : f32
        %6 = math.tanh %5 : f32
        %7 = arith.negf %6 : f32
        %8 = arith.mulf %7, %b : f32
        %9 = arith.divf %8, %c : f32
        %10 = arith.addf %9, %c : f32
        %11 = arith.subf %10, %o : f32
        %12 = arith.extf %11 : f32 to f64
        %13 = arith.bitcast %12 : f64 to i64
        %14 = arith.addi %13, %13 : i64
        %15 = arith.andi %14, %i : i64
        %16 = arith.trunci %15 : i64 to i16
        %17 = arith.extsi %16 : i16 to i32
        %18 = arith.uitofp %17 : i32 to f32
        linalg.yield %18 : f32
  } -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

