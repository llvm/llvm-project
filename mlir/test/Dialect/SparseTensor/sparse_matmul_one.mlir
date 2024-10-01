// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --sparsification-and-bufferization | FileCheck %s

#CSR_ones_complex = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  explicitVal = #complex.number<:f32 1.0, 0.0>,
  implicitVal = #complex.number<:f32 0.0, 0.0>
}>

#CSR_ones_fp = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  explicitVal = 1.0 : f32,
  implicitVal = 0.0 : f32
}>

#CSR_ones_int = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  explicitVal = 1 : i32,
  implicitVal = 0 : i32
}>

// CHECK-LABEL:   func.func @matmul_complex
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[X:.*]] = memref.load
// CHECK:             scf.for
// CHECK:               %[[I:.*]] = memref.load
// CHECK:               %[[Y:.*]] = memref.load
// CHECK:               %[[M:.*]] = complex.add %[[Y]], %[[X]] : complex<f32>
// CHECK:               memref.store %[[M]]
// CHECK:             }
// CHECK:           }
// CHECK:         }
func.func @matmul_complex(%a: tensor<10x20xcomplex<f32>>,
                          %b: tensor<20x30xcomplex<f32>, #CSR_ones_complex>,
                          %c: tensor<10x30xcomplex<f32>>) -> tensor<10x30xcomplex<f32>> {
  %0 = linalg.matmul
    ins(%a, %b: tensor<10x20xcomplex<f32>>, tensor<20x30xcomplex<f32>,#CSR_ones_complex>)
    outs(%c: tensor<10x30xcomplex<f32>>) -> tensor<10x30xcomplex<f32>>
  return %0 : tensor<10x30xcomplex<f32>>
}

// CHECK-LABEL:   func.func @matmul_fp
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[X:.*]] = memref.load
// CHECK:             scf.for
// CHECK:               %[[I:.*]] = memref.load
// CHECK:               %[[Y:.*]] = memref.load
// CHECK:               %[[M:.*]] = arith.addf %[[Y]], %[[X]] : f32
// CHECK:               memref.store %[[M]]
// CHECK:             }
// CHECK:           }
// CHECK:         }
func.func @matmul_fp(%a: tensor<10x20xf32>,
                     %b: tensor<20x30xf32, #CSR_ones_fp>,
                     %c: tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul
    ins(%a, %b: tensor<10x20xf32>, tensor<20x30xf32,#CSR_ones_fp>)
    outs(%c: tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}

// CHECK-LABEL:   func.func @matmul_int
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[X:.*]] = memref.load
// CHECK:             scf.for
// CHECK:               %[[I:.*]] = memref.load
// CHECK:               %[[Y:.*]] = memref.load
// CHECK:               %[[M:.*]] = arith.addi %[[Y]], %[[X]] : i32
// CHECK:               memref.store %[[M]]
// CHECK:             }
// CHECK:           }
// CHECK:         }
func.func @matmul_int(%a: tensor<10x20xi32>,
                      %b: tensor<20x30xi32, #CSR_ones_int>,
                      %c: tensor<10x30xi32>) -> tensor<10x30xi32> {
  %0 = linalg.matmul
    ins(%a, %b: tensor<10x20xi32>, tensor<20x30xi32,#CSR_ones_int>)
    outs(%c: tensor<10x30xi32>) -> tensor<10x30xi32>
  return %0 : tensor<10x30xi32>
}
