
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,canonicalize,linalg-generalize-named-ops,tosa-to-arith,tosa-to-tensor,canonicalize))" %s -o %t.model.linalg.mlir
// RUN: mlir-opt --canonicalize --linalg-fuse-elementwise-ops --linalg-inline-scalar-operands --linalg-fold-unit-extent-dims --fold-tensor-subset-ops --canonicalize %t.model.linalg.mlir -o %t.model.linalg.opt.mlir
// RUN: mlir-opt --pass-pipeline='builtin.module(one-shot-bufferize{allow-unknown-ops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}, canonicalize)' %t.model.linalg.opt.mlir -o %t.model.buffers.mlir

// RUN: mlir-opt --canonicalize  --buffer-results-to-out-params --buffer-hoisting --buffer-loop-hoisting --promote-buffers-to-stack --fold-memref-alias-ops --canonicalize  --buffer-deallocation-pipeline --canonicalize %t.model.buffers.mlir -o %t.model.buffers.opt.mlir
// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --fold-memref-alias-ops --canonicalize %t.model.buffers.opt.mlir -o %t.model.scf.mlir

// RUN: python %S/fix_mem.py -p memref-copy %t.model.scf.mlir %t.model.scf.1.mlir

// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --canonicalize %t.model.scf.1.mlir -o %t.model.scf.2.mlir
// RUN: mlir-opt --canonicalize --fold-memref-alias-ops --normalize-memrefs --canonicalize %t.model.scf.2.mlir -o %t.model.scf.3.mlir

// RUN: mlir-opt --arith-expand --canonicalize %t.model.scf.3.mlir -o %t.model.scf.4.mlir

// RUN: python %S/fix_mem.py %t.model.scf.4.mlir %t.model.scf.5.mlir

// RUN: mlir-opt --convert-math-to-libm --canonicalize %t.model.scf.5.mlir -o %t.model.scf.6.mlir
// RUN: mlir-opt --convert-func-to-emitc --convert-scf-to-emitc --convert-arith-to-emitc --convert-memref-to-emitc --canonicalize %t.model.scf.6.mlir -o %t.model.emitc.mlir

// RUN: mlir-translate --mlir-to-cpp %t.model.emitc.mlir | FileCheck %s

// CHECK: Fail this test

// -----


module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<f32> {ml_program.identifier = "serve_b:0", tf_saved_model.index_path = ["b"]}, %arg1: tensor<f32> {ml_program.identifier = "serve_a:0", tf_saved_model.index_path = ["a"]}) -> (tensor<f32> {ml_program.identifier = "PartitionedCall:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serve"]} {
    %0 = tosa.add %arg1, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.mul %arg1, %arg0 {shift = 0 : i8} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = tosa.add %1, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = tosa.add %2, %1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %4 = tosa.add %arg1, %3 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = tosa.mul %1, %arg1 {shift = 0 : i8} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %6 = tosa.sub %5, %4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = tosa.reciprocal %5 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.mul %6, %7 {shift = 0 : i8} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = tosa.sub %1, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = tosa.add %0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = tosa.add %arg0, %10 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %12 = tosa.add %9, %11 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %13 = tosa.add %1, %12 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %14 = tosa.add %8, %13 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %14 : tensor<f32>
  }
}


