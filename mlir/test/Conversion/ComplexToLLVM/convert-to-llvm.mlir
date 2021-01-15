// RUN: mlir-opt %s -split-input-file -convert-complex-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @complex_numbers()
// CHECK-NEXT:    %[[REAL0:.*]] = llvm.mlir.constant(1.200000e+00 : f32) : f32
// CHECK-NEXT:    %[[IMAG0:.*]] = llvm.mlir.constant(3.400000e+00 : f32) : f32
// CHECK-NEXT:    %[[CPLX0:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[CPLX1:.*]] = llvm.insertvalue %[[REAL0]], %[[CPLX0]][0] : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[CPLX2:.*]] = llvm.insertvalue %[[IMAG0]], %[[CPLX1]][1] : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[REAL1:.*]] = llvm.extractvalue %[[CPLX2:.*]][0] : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[IMAG1:.*]] = llvm.extractvalue %[[CPLX2:.*]][1] : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    llvm.return
func @complex_numbers() {
  %real0 = constant 1.2 : f32
  %imag0 = constant 3.4 : f32
  %cplx2 = complex.create %real0, %imag0 : complex<f32>
  %real1 = complex.re%cplx2 : complex<f32>
  %imag1 = complex.im %cplx2 : complex<f32>
  return
}

// CHECK-LABEL: llvm.func @complex_addition()
// CHECK-DAG:     %[[A_REAL:.*]] = llvm.extractvalue %[[A:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_REAL:.*]] = llvm.extractvalue %[[B:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[A_IMAG:.*]] = llvm.extractvalue %[[A]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_IMAG:.*]] = llvm.extractvalue %[[B]][1] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C0:.*]] = llvm.mlir.undef : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[C_REAL:.*]] = llvm.fadd %[[A_REAL]], %[[B_REAL]] : f64
// CHECK-DAG:     %[[C_IMAG:.*]] = llvm.fadd %[[A_IMAG]], %[[B_IMAG]] : f64
// CHECK:         %[[C1:.*]] = llvm.insertvalue %[[C_REAL]], %[[C0]][0] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C2:.*]] = llvm.insertvalue %[[C_IMAG]], %[[C1]][1] : !llvm.struct<(f64, f64)>
func @complex_addition() {
  %a_re = constant 1.2 : f64
  %a_im = constant 3.4 : f64
  %a = complex.create %a_re, %a_im : complex<f64>
  %b_re = constant 5.6 : f64
  %b_im = constant 7.8 : f64
  %b = complex.create %b_re, %b_im : complex<f64>
  %c = complex.add %a, %b : complex<f64>
  return
}

// CHECK-LABEL: llvm.func @complex_substraction()
// CHECK-DAG:     %[[A_REAL:.*]] = llvm.extractvalue %[[A:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_REAL:.*]] = llvm.extractvalue %[[B:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[A_IMAG:.*]] = llvm.extractvalue %[[A]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_IMAG:.*]] = llvm.extractvalue %[[B]][1] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C0:.*]] = llvm.mlir.undef : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[C_REAL:.*]] = llvm.fsub %[[A_REAL]], %[[B_REAL]] : f64
// CHECK-DAG:     %[[C_IMAG:.*]] = llvm.fsub %[[A_IMAG]], %[[B_IMAG]] : f64
// CHECK:         %[[C1:.*]] = llvm.insertvalue %[[C_REAL]], %[[C0]][0] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C2:.*]] = llvm.insertvalue %[[C_IMAG]], %[[C1]][1] : !llvm.struct<(f64, f64)>
func @complex_substraction() {
  %a_re = constant 1.2 : f64
  %a_im = constant 3.4 : f64
  %a = complex.create %a_re, %a_im : complex<f64>
  %b_re = constant 5.6 : f64
  %b_im = constant 7.8 : f64
  %b = complex.create %b_re, %b_im : complex<f64>
  %c = complex.sub %a, %b : complex<f64>
  return
}
