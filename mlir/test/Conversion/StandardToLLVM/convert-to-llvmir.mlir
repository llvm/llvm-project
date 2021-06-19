// RUN: mlir-opt -convert-std-to-llvm %s -split-input-file | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='index-bitwidth=32' %s -split-input-file | FileCheck --check-prefix=CHECK32 %s

// CHECK-LABEL: func @empty() {
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }
func @empty() {
^bb0:
  return
}

// CHECK-LABEL: llvm.func @body(i64)
func private @body(index)

// CHECK-LABEL: func @simple_loop() {
// CHECK32-LABEL: func @simple_loop() {
func @simple_loop() {
^bb0:
// CHECK-NEXT:  llvm.br ^bb1
// CHECK32-NEXT:  llvm.br ^bb1
  br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK32-NEXT: ^bb1:	// pred: ^bb0
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i32
// CHECK32-NEXT:  llvm.br ^bb2({{.*}} : i32)
^bb1:	// pred: ^bb0
  %c1 = constant 1 : index
  %c42 = constant 42 : index
  br ^bb2(%c1 : index)

// CHECK:      ^bb2({{.*}}: i64):	// 2 preds: ^bb1, ^bb3
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
// CHECK32:      ^bb2({{.*}}: i32):	// 2 preds: ^bb1, ^bb3
// CHECK32-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i32
// CHECK32-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = cmpi slt, %0, %c42 : index
  cond_br %1, ^bb3, ^bb4

// CHECK:      ^bb3:	// pred: ^bb2
// CHECK-NEXT:  llvm.call @body({{.*}}) : (i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK32:      ^bb3:	// pred: ^bb2
// CHECK32-NEXT:  llvm.call @body({{.*}}) : (i32) -> ()
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i32
// CHECK32-NEXT:  llvm.br ^bb2({{.*}} : i32)
^bb3:	// pred: ^bb2
  call @body(%0) : (index) -> ()
  %c1_0 = constant 1 : index
  %2 = addi %0, %c1_0 : index
  br ^bb2(%2 : index)

// CHECK:      ^bb4:	// pred: ^bb2
// CHECK-NEXT:  llvm.return
^bb4:	// pred: ^bb2
  return
}

// CHECK-LABEL: func @simple_caller() {
// CHECK-NEXT:  llvm.call @simple_loop() : () -> ()
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }
func @simple_caller() {
^bb0:
  call @simple_loop() : () -> ()
  return
}

// Check that function call attributes persist during conversion.
// CHECK-LABEL: @call_with_attributes
func @call_with_attributes() {
  // CHECK: llvm.call @simple_loop() {baz = [1, 2, 3, 4], foo = "bar"} : () -> ()
  call @simple_loop() {foo="bar", baz=[1,2,3,4]} : () -> ()
  return
}

// CHECK-LABEL: func @ml_caller() {
// CHECK-NEXT:  llvm.call @simple_loop() : () -> ()
// CHECK-NEXT:  llvm.call @more_imperfectly_nested_loops() : () -> ()
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }
func @ml_caller() {
^bb0:
  call @simple_loop() : () -> ()
  call @more_imperfectly_nested_loops() : () -> ()
  return
}

// CHECK-LABEL: llvm.func @body_args(i64) -> i64
// CHECK32-LABEL: llvm.func @body_args(i32) -> i32
func private @body_args(index) -> index
// CHECK-LABEL: llvm.func @other(i64, i32) -> i32
// CHECK32-LABEL: llvm.func @other(i32, i32) -> i32
func private @other(index, i32) -> i32

// CHECK-LABEL: func @func_args(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  llvm.br ^bb1
// CHECK32-LABEL: func @func_args(%arg0: i32, %arg1: i32) -> i32 {
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(0 : i32) : i32
// CHECK32-NEXT:  llvm.br ^bb1
func @func_args(i32, i32) -> i32 {
^bb0(%arg0: i32, %arg1: i32):
  %c0_i32 = constant 0 : i32
  br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK32-NEXT: ^bb1:	// pred: ^bb0
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i32
// CHECK32-NEXT:  llvm.br ^bb2({{.*}} : i32)
^bb1:	// pred: ^bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br ^bb2(%c0 : index)

// CHECK-NEXT: ^bb2({{.*}}: i64):	// 2 preds: ^bb1, ^bb3
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
// CHECK32-NEXT: ^bb2({{.*}}: i32):	// 2 preds: ^bb1, ^bb3
// CHECK32-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i32
// CHECK32-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = cmpi slt, %0, %c42 : index
  cond_br %1, ^bb3, ^bb4

// CHECK-NEXT: ^bb3:	// pred: ^bb2
// CHECK-NEXT:  {{.*}} = llvm.call @body_args({{.*}}) : (i64) -> i64
// CHECK-NEXT:  {{.*}} = llvm.call @other({{.*}}, %arg0) : (i64, i32) -> i32
// CHECK-NEXT:  {{.*}} = llvm.call @other({{.*}}, {{.*}}) : (i64, i32) -> i32
// CHECK-NEXT:  {{.*}} = llvm.call @other({{.*}}, %arg1) : (i64, i32) -> i32
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK32-NEXT: ^bb3:	// pred: ^bb2
// CHECK32-NEXT:  {{.*}} = llvm.call @body_args({{.*}}) : (i32) -> i32
// CHECK32-NEXT:  {{.*}} = llvm.call @other({{.*}}, %arg0) : (i32, i32) -> i32
// CHECK32-NEXT:  {{.*}} = llvm.call @other({{.*}}, {{.*}}) : (i32, i32) -> i32
// CHECK32-NEXT:  {{.*}} = llvm.call @other({{.*}}, %arg1) : (i32, i32) -> i32
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i32
// CHECK32-NEXT:  llvm.br ^bb2({{.*}} : i32)
^bb3:	// pred: ^bb2
  %2 = call @body_args(%0) : (index) -> index
  %3 = call @other(%2, %arg0) : (index, i32) -> i32
  %4 = call @other(%2, %3) : (index, i32) -> i32
  %5 = call @other(%2, %arg1) : (index, i32) -> i32
  %c1 = constant 1 : index
  %6 = addi %0, %c1 : index
  br ^bb2(%6 : index)

// CHECK-NEXT: ^bb4:	// pred: ^bb2
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.call @other({{.*}}, {{.*}}) : (i64, i32) -> i32
// CHECK-NEXT:  llvm.return {{.*}} : i32
// CHECK32-NEXT: ^bb4:	// pred: ^bb2
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.call @other({{.*}}, {{.*}}) : (i32, i32) -> i32
// CHECK32-NEXT:  llvm.return {{.*}} : i32
^bb4:	// pred: ^bb2
  %c0_0 = constant 0 : index
  %7 = call @other(%c0_0, %c0_i32) : (index, i32) -> i32
  return %7 : i32
}

// CHECK-LABEL: llvm.func @pre(i64)
// CHECK32-LABEL: llvm.func @pre(i32)
func private @pre(index)

// CHECK-LABEL: llvm.func @body2(i64, i64)
// CHECK32-LABEL: llvm.func @body2(i32, i32)
func private @body2(index, index)

// CHECK-LABEL: llvm.func @post(i64)
// CHECK32-LABEL: llvm.func @post(i32)
func private @post(index)

// CHECK-LABEL: func @imperfectly_nested_loops() {
// CHECK-NEXT:  llvm.br ^bb1
func @imperfectly_nested_loops() {
^bb0:
  br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
^bb1:	// pred: ^bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br ^bb2(%c0 : index)

// CHECK-NEXT: ^bb2({{.*}}: i64):	// 2 preds: ^bb1, ^bb7
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb8
^bb2(%0: index):	// 2 preds: ^bb1, ^bb7
  %1 = cmpi slt, %0, %c42 : index
  cond_br %1, ^bb3, ^bb8

// CHECK-NEXT: ^bb3:
// CHECK-NEXT:  llvm.call @pre({{.*}}) : (i64) -> ()
// CHECK-NEXT:  llvm.br ^bb4
^bb3:	// pred: ^bb2
  call @pre(%0) : (index) -> ()
  br ^bb4

// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(7 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(56 : index) : i64
// CHECK-NEXT:  llvm.br ^bb5({{.*}} : i64)
^bb4:	// pred: ^bb3
  %c7 = constant 7 : index
  %c56 = constant 56 : index
  br ^bb5(%c7 : index)

// CHECK-NEXT: ^bb5({{.*}}: i64):	// 2 preds: ^bb4, ^bb6
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb6, ^bb7
^bb5(%2: index):	// 2 preds: ^bb4, ^bb6
  %3 = cmpi slt, %2, %c56 : index
  cond_br %3, ^bb6, ^bb7

// CHECK-NEXT: ^bb6:	// pred: ^bb5
// CHECK-NEXT:  llvm.call @body2({{.*}}, {{.*}}) : (i64, i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb5({{.*}} : i64)
^bb6:	// pred: ^bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = constant 2 : index
  %4 = addi %2, %c2 : index
  br ^bb5(%4 : index)

// CHECK-NEXT: ^bb7:	// pred: ^bb5
// CHECK-NEXT:  llvm.call @post({{.*}}) : (i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
^bb7:	// pred: ^bb5
  call @post(%0) : (index) -> ()
  %c1 = constant 1 : index
  %5 = addi %0, %c1 : index
  br ^bb2(%5 : index)

// CHECK-NEXT: ^bb8:	// pred: ^bb2
// CHECK-NEXT:  llvm.return
^bb8:	// pred: ^bb2
  return
}

// CHECK-LABEL: llvm.func @mid(i64)
func private @mid(index)

// CHECK-LABEL: llvm.func @body3(i64, i64)
func private @body3(index, index)

// A complete function transformation check.
// CHECK-LABEL: func @more_imperfectly_nested_loops() {
// CHECK-NEXT:  llvm.br ^bb1
// CHECK-NEXT:^bb1:	// pred: ^bb0
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK-NEXT:^bb2({{.*}}: i64):	// 2 preds: ^bb1, ^bb11
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb12
// CHECK-NEXT:^bb3:	// pred: ^bb2
// CHECK-NEXT:  llvm.call @pre({{.*}}) : (i64) -> ()
// CHECK-NEXT:  llvm.br ^bb4
// CHECK-NEXT:^bb4:	// pred: ^bb3
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(7 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(56 : index) : i64
// CHECK-NEXT:  llvm.br ^bb5({{.*}} : i64)
// CHECK-NEXT:^bb5({{.*}}: i64):	// 2 preds: ^bb4, ^bb6
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb6, ^bb7
// CHECK-NEXT:^bb6:	// pred: ^bb5
// CHECK-NEXT:  llvm.call @body2({{.*}}, {{.*}}) : (i64, i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb5({{.*}} : i64)
// CHECK-NEXT:^bb7:	// pred: ^bb5
// CHECK-NEXT:  llvm.call @mid({{.*}}) : (i64) -> ()
// CHECK-NEXT:  llvm.br ^bb8
// CHECK-NEXT:^bb8:	// pred: ^bb7
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(18 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(37 : index) : i64
// CHECK-NEXT:  llvm.br ^bb9({{.*}} : i64)
// CHECK-NEXT:^bb9({{.*}}: i64):	// 2 preds: ^bb8, ^bb10
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb10, ^bb11
// CHECK-NEXT:^bb10:	// pred: ^bb9
// CHECK-NEXT:  llvm.call @body3({{.*}}, {{.*}}) : (i64, i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb9({{.*}} : i64)
// CHECK-NEXT:^bb11:	// pred: ^bb9
// CHECK-NEXT:  llvm.call @post({{.*}}) : (i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK-NEXT:^bb12:	// pred: ^bb2
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }
func @more_imperfectly_nested_loops() {
^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  br ^bb2(%c0 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb11
  %1 = cmpi slt, %0, %c42 : index
  cond_br %1, ^bb3, ^bb12
^bb3:	// pred: ^bb2
  call @pre(%0) : (index) -> ()
  br ^bb4
^bb4:	// pred: ^bb3
  %c7 = constant 7 : index
  %c56 = constant 56 : index
  br ^bb5(%c7 : index)
^bb5(%2: index):	// 2 preds: ^bb4, ^bb6
  %3 = cmpi slt, %2, %c56 : index
  cond_br %3, ^bb6, ^bb7
^bb6:	// pred: ^bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = constant 2 : index
  %4 = addi %2, %c2 : index
  br ^bb5(%4 : index)
^bb7:	// pred: ^bb5
  call @mid(%0) : (index) -> ()
  br ^bb8
^bb8:	// pred: ^bb7
  %c18 = constant 18 : index
  %c37 = constant 37 : index
  br ^bb9(%c18 : index)
^bb9(%5: index):	// 2 preds: ^bb8, ^bb10
  %6 = cmpi slt, %5, %c37 : index
  cond_br %6, ^bb10, ^bb11
^bb10:	// pred: ^bb9
  call @body3(%0, %5) : (index, index) -> ()
  %c3 = constant 3 : index
  %7 = addi %5, %c3 : index
  br ^bb9(%7 : index)
^bb11:	// pred: ^bb9
  call @post(%0) : (index) -> ()
  %c1 = constant 1 : index
  %8 = addi %0, %c1 : index
  br ^bb2(%8 : index)
^bb12:	// pred: ^bb2
  return
}

// CHECK-LABEL: llvm.func @get_i64() -> i64
func private @get_i64() -> (i64)
// CHECK-LABEL: llvm.func @get_f32() -> f32
func private @get_f32() -> (f32)
// CHECK-LABEL: llvm.func @get_c16() -> !llvm.struct<(f16, f16)>
func private @get_c16() -> (complex<f16>)
// CHECK-LABEL: llvm.func @get_c32() -> !llvm.struct<(f32, f32)>
func private @get_c32() -> (complex<f32>)
// CHECK-LABEL: llvm.func @get_c64() -> !llvm.struct<(f64, f64)>
func private @get_c64() -> (complex<f64>)
// CHECK-LABEL: llvm.func @get_memref() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
// CHECK32-LABEL: llvm.func @get_memref() -> !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>
func private @get_memref() -> (memref<42x?x10x?xf32>)

// CHECK-LABEL: llvm.func @multireturn() -> !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)> {
// CHECK32-LABEL: llvm.func @multireturn() -> !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)> {
func @multireturn() -> (i64, f32, memref<42x?x10x?xf32>) {
^bb0:
// CHECK-NEXT:  {{.*}} = llvm.call @get_i64() : () -> i64
// CHECK-NEXT:  {{.*}} = llvm.call @get_f32() : () -> f32
// CHECK-NEXT:  {{.*}} = llvm.call @get_memref() : () -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
// CHECK32-NEXT:  {{.*}} = llvm.call @get_i64() : () -> i64
// CHECK32-NEXT:  {{.*}} = llvm.call @get_f32() : () -> f32
// CHECK32-NEXT:  {{.*}} = llvm.call @get_memref() : () -> !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>
  %0 = call @get_i64() : () -> (i64)
  %1 = call @get_f32() : () -> (f32)
  %2 = call @get_memref() : () -> (memref<42x?x10x?xf32>)
// CHECK-NEXT:  {{.*}} = llvm.mlir.undef : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  llvm.return {{.*}} : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.mlir.undef : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  llvm.return {{.*}} : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
  return %0, %1, %2 : i64, f32, memref<42x?x10x?xf32>
}


// CHECK-LABEL: llvm.func @multireturn_caller() {
// CHECK32-LABEL: llvm.func @multireturn_caller() {
func @multireturn_caller() {
^bb0:
// CHECK-NEXT:  {{.*}} = llvm.call @multireturn() : () -> !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.call @multireturn() : () -> !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i32, array<4 x i32>, array<4 x i32>)>)>
  %0:3 = call @multireturn() : () -> (i64, f32, memref<42x?x10x?xf32>)
  %1 = constant 42 : i64
// CHECK:       {{.*}} = llvm.add {{.*}}, {{.*}} : i64
  %2 = addi %0#0, %1 : i64
  %3 = constant 42.0 : f32
// CHECK:       {{.*}} = llvm.fadd {{.*}}, {{.*}} : f32
  %4 = addf %0#1, %3 : f32
  %5 = constant 0 : index
  return
}

// CHECK-LABEL: llvm.func @vector_ops(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: vector<4xi64>, %arg3: vector<4xi64>) -> vector<4xf32> {
func @vector_ops(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: vector<4xi64>, %arg3: vector<4xi64>) -> vector<4xf32> {
// CHECK-NEXT:  %0 = llvm.mlir.constant(dense<4.200000e+01> : vector<4xf32>) : vector<4xf32>
  %0 = constant dense<42.> : vector<4xf32>
// CHECK-NEXT:  %1 = llvm.fadd %arg0, %0 : vector<4xf32>
  %1 = addf %arg0, %0 : vector<4xf32>
// CHECK-NEXT:  %2 = llvm.sdiv %arg2, %arg2 : vector<4xi64>
  %3 = divi_signed %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:  %3 = llvm.udiv %arg2, %arg2 : vector<4xi64>
  %4 = divi_unsigned %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:  %4 = llvm.srem %arg2, %arg2 : vector<4xi64>
  %5 = remi_signed %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:  %5 = llvm.urem %arg2, %arg2 : vector<4xi64>
  %6 = remi_unsigned %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:  %6 = llvm.fdiv %arg0, %0 : vector<4xf32>
  %7 = divf %arg0, %0 : vector<4xf32>
// CHECK-NEXT:  %7 = llvm.frem %arg0, %0 : vector<4xf32>
  %8 = remf %arg0, %0 : vector<4xf32>
// CHECK-NEXT:  %8 = llvm.and %arg2, %arg3 : vector<4xi64>
  %9 = and %arg2, %arg3 : vector<4xi64>
// CHECK-NEXT:  %9 = llvm.or %arg2, %arg3 : vector<4xi64>
  %10 = or %arg2, %arg3 : vector<4xi64>
// CHECK-NEXT:  %10 = llvm.xor %arg2, %arg3 : vector<4xi64>
  %11 = xor %arg2, %arg3 : vector<4xi64>
// CHECK-NEXT:  %11 = llvm.shl %arg2, %arg2 : vector<4xi64>
  %12 = shift_left %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:  %12 = llvm.ashr %arg2, %arg2 : vector<4xi64>
  %13 = shift_right_signed %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:  %13 = llvm.lshr %arg2, %arg2 : vector<4xi64>
  %14 = shift_right_unsigned %arg2, %arg2 : vector<4xi64>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @ops
func @ops(f32, f32, i32, i32, f64) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: f64):
// CHECK-NEXT:  %0 = llvm.fsub %arg0, %arg1 : f32
  %0 = subf %arg0, %arg1: f32
// CHECK-NEXT:  %1 = llvm.sub %arg2, %arg3 : i32
  %1 = subi %arg2, %arg3: i32
// CHECK-NEXT:  %2 = llvm.icmp "slt" %arg2, %1 : i32
  %2 = cmpi slt, %arg2, %1 : i32
// CHECK-NEXT:  %3 = llvm.sdiv %arg2, %arg3 : i32
  %3 = divi_signed %arg2, %arg3 : i32
// CHECK-NEXT:  %4 = llvm.udiv %arg2, %arg3 : i32
  %4 = divi_unsigned %arg2, %arg3 : i32
// CHECK-NEXT:  %5 = llvm.srem %arg2, %arg3 : i32
  %5 = remi_signed %arg2, %arg3 : i32
// CHECK-NEXT:  %6 = llvm.urem %arg2, %arg3 : i32
  %6 = remi_unsigned %arg2, %arg3 : i32
// CHECK-NEXT:  %7 = llvm.select %2, %arg2, %arg3 : i1, i32
  %7 = select %2, %arg2, %arg3 : i32
// CHECK-NEXT:  %8 = llvm.fdiv %arg0, %arg1 : f32
  %8 = divf %arg0, %arg1 : f32
// CHECK-NEXT:  %9 = llvm.frem %arg0, %arg1 : f32
  %9 = remf %arg0, %arg1 : f32
// CHECK-NEXT: %10 = llvm.and %arg2, %arg3 : i32
  %10 = and %arg2, %arg3 : i32
// CHECK-NEXT: %11 = llvm.or %arg2, %arg3 : i32
  %11 = or %arg2, %arg3 : i32
// CHECK-NEXT: %12 = llvm.xor %arg2, %arg3 : i32
  %12 = xor %arg2, %arg3 : i32
// CHECK-NEXT: %13 = "llvm.intr.exp"(%arg0) : (f32) -> f32
  %13 = math.exp %arg0 : f32
// CHECK-NEXT: %14 = "llvm.intr.exp2"(%arg0) : (f32) -> f32
  %14 = math.exp2 %arg0 : f32
// CHECK-NEXT: %15 = llvm.mlir.constant(7.900000e-01 : f64) : f64
  %15 = constant 7.9e-01 : f64
// CHECK-NEXT: %16 = llvm.shl %arg2, %arg3 : i32
  %16 = shift_left %arg2, %arg3 : i32
// CHECK-NEXT: %17 = llvm.ashr %arg2, %arg3 : i32
  %17 = shift_right_signed %arg2, %arg3 : i32
// CHECK-NEXT: %18 = llvm.lshr %arg2, %arg3 : i32
  %18 = shift_right_unsigned %arg2, %arg3 : i32
// CHECK-NEXT: %{{[0-9]+}} = "llvm.intr.sqrt"(%arg0) : (f32) -> f32
  %19 = math.sqrt %arg0 : f32
// CHECK-NEXT: %{{[0-9]+}} = "llvm.intr.sqrt"(%arg4) : (f64) -> f64
  %20 = math.sqrt %arg4 : f64
  return %0, %4 : f32, i32
}

// Checking conversion of index types to integers using i1, assuming no target
// system would have a 1-bit address space.  Otherwise, we would have had to
// make this test dependent on the pointer size on the target system.
// CHECK-LABEL: @index_cast
func @index_cast(%arg0: index, %arg1: i1) {
// CHECK-NEXT: = llvm.trunc %arg0 : i{{.*}} to i1
  %0 = index_cast %arg0: index to i1
// CHECK-NEXT: = llvm.sext %arg1 : i1 to i{{.*}}
  %1 = index_cast %arg1: i1 to index
  return
}

// CHECK-LABEL: @vector_index_cast
func @vector_index_cast(%arg0: vector<2xindex>, %arg1: vector<2xi1>) {
// CHECK-NEXT: = llvm.trunc %{{.*}} : vector<2xi{{.*}}> to vector<2xi1>
  %0 = index_cast %arg0: vector<2xindex> to vector<2xi1>
// CHECK-NEXT: = llvm.sext %{{.*}} : vector<2xi1> to vector<2xi{{.*}}>
  %1 = index_cast %arg1: vector<2xi1> to vector<2xindex>
  return
}

// Checking conversion of signed integer types to floating point.
// CHECK-LABEL: @sitofp
func @sitofp(%arg0 : i32, %arg1 : i64) {
// CHECK-NEXT: = llvm.sitofp {{.*}} : i32 to f32
  %0 = sitofp %arg0: i32 to f32
// CHECK-NEXT: = llvm.sitofp {{.*}} : i32 to f64
  %1 = sitofp %arg0: i32 to f64
// CHECK-NEXT: = llvm.sitofp {{.*}} : i64 to f32
  %2 = sitofp %arg1: i64 to f32
// CHECK-NEXT: = llvm.sitofp {{.*}} : i64 to f64
  %3 = sitofp %arg1: i64 to f64
  return
}

// Checking conversion of integer vectors to floating point vector types.
// CHECK-LABEL: @sitofp_vector
func @sitofp_vector(%arg0 : vector<2xi16>, %arg1 : vector<2xi32>, %arg2 : vector<2xi64>) {
// CHECK-NEXT: = llvm.sitofp {{.*}} : vector<2xi16> to vector<2xf32>
  %0 = sitofp %arg0: vector<2xi16> to vector<2xf32>
// CHECK-NEXT: = llvm.sitofp {{.*}} : vector<2xi16> to vector<2xf64>
  %1 = sitofp %arg0: vector<2xi16> to vector<2xf64>
// CHECK-NEXT: = llvm.sitofp {{.*}} : vector<2xi32> to vector<2xf32>
  %2 = sitofp %arg1: vector<2xi32> to vector<2xf32>
// CHECK-NEXT: = llvm.sitofp {{.*}} : vector<2xi32> to vector<2xf64>
  %3 = sitofp %arg1: vector<2xi32> to vector<2xf64>
// CHECK-NEXT: = llvm.sitofp {{.*}} : vector<2xi64> to vector<2xf32>
  %4 = sitofp %arg2: vector<2xi64> to vector<2xf32>
// CHECK-NEXT: = llvm.sitofp {{.*}} : vector<2xi64> to vector<2xf64>
  %5 = sitofp %arg2: vector<2xi64> to vector<2xf64>
  return
}

// Checking conversion of unsigned integer types to floating point.
// CHECK-LABEL: @uitofp
func @uitofp(%arg0 : i32, %arg1 : i64) {
// CHECK-NEXT: = llvm.uitofp {{.*}} : i32 to f32
  %0 = uitofp %arg0: i32 to f32
// CHECK-NEXT: = llvm.uitofp {{.*}} : i32 to f64
  %1 = uitofp %arg0: i32 to f64
// CHECK-NEXT: = llvm.uitofp {{.*}} : i64 to f32
  %2 = uitofp %arg1: i64 to f32
// CHECK-NEXT: = llvm.uitofp {{.*}} : i64 to f64
  %3 = uitofp %arg1: i64 to f64
  return
}

// Checking conversion of integer types to floating point.
// CHECK-LABEL: @fpext
func @fpext(%arg0 : f16, %arg1 : f32) {
// CHECK-NEXT: = llvm.fpext {{.*}} : f16 to f32
  %0 = fpext %arg0: f16 to f32
// CHECK-NEXT: = llvm.fpext {{.*}} : f16 to f64
  %1 = fpext %arg0: f16 to f64
// CHECK-NEXT: = llvm.fpext {{.*}} : f32 to f64
  %2 = fpext %arg1: f32 to f64
  return
}

// Checking conversion of integer types to floating point.
// CHECK-LABEL: @fpext
func @fpext_vector(%arg0 : vector<2xf16>, %arg1 : vector<2xf32>) {
// CHECK-NEXT: = llvm.fpext {{.*}} : vector<2xf16> to vector<2xf32>
  %0 = fpext %arg0: vector<2xf16> to vector<2xf32>
// CHECK-NEXT: = llvm.fpext {{.*}} : vector<2xf16> to vector<2xf64>
  %1 = fpext %arg0: vector<2xf16> to vector<2xf64>
// CHECK-NEXT: = llvm.fpext {{.*}} : vector<2xf32> to vector<2xf64>
  %2 = fpext %arg1: vector<2xf32> to vector<2xf64>
  return
}

// Checking conversion of floating point to integer types.
// CHECK-LABEL: @fptosi
func @fptosi(%arg0 : f32, %arg1 : f64) {
// CHECK-NEXT: = llvm.fptosi {{.*}} : f32 to i32
  %0 = fptosi %arg0: f32 to i32
// CHECK-NEXT: = llvm.fptosi {{.*}} : f32 to i64
  %1 = fptosi %arg0: f32 to i64
// CHECK-NEXT: = llvm.fptosi {{.*}} : f64 to i32
  %2 = fptosi %arg1: f64 to i32
// CHECK-NEXT: = llvm.fptosi {{.*}} : f64 to i64
  %3 = fptosi %arg1: f64 to i64
  return
}

// Checking conversion of floating point vectors to integer vector types.
// CHECK-LABEL: @fptosi_vector
func @fptosi_vector(%arg0 : vector<2xf16>, %arg1 : vector<2xf32>, %arg2 : vector<2xf64>) {
// CHECK-NEXT: = llvm.fptosi {{.*}} : vector<2xf16> to vector<2xi32>
  %0 = fptosi %arg0: vector<2xf16> to vector<2xi32>
// CHECK-NEXT: = llvm.fptosi {{.*}} : vector<2xf16> to vector<2xi64>
  %1 = fptosi %arg0: vector<2xf16> to vector<2xi64>
// CHECK-NEXT: = llvm.fptosi {{.*}} : vector<2xf32> to vector<2xi32>
  %2 = fptosi %arg1: vector<2xf32> to vector<2xi32>
// CHECK-NEXT: = llvm.fptosi {{.*}} : vector<2xf32> to vector<2xi64>
  %3 = fptosi %arg1: vector<2xf32> to vector<2xi64>
// CHECK-NEXT: = llvm.fptosi {{.*}} : vector<2xf64> to vector<2xi32>
  %4 = fptosi %arg2: vector<2xf64> to vector<2xi32>
// CHECK-NEXT: = llvm.fptosi {{.*}} : vector<2xf64> to vector<2xi64>
  %5 = fptosi %arg2: vector<2xf64> to vector<2xi64>
  return
}

// Checking conversion of floating point to integer types.
// CHECK-LABEL: @fptoui
func @fptoui(%arg0 : f32, %arg1 : f64) {
// CHECK-NEXT: = llvm.fptoui {{.*}} : f32 to i32
  %0 = fptoui %arg0: f32 to i32
// CHECK-NEXT: = llvm.fptoui {{.*}} : f32 to i64
  %1 = fptoui %arg0: f32 to i64
// CHECK-NEXT: = llvm.fptoui {{.*}} : f64 to i32
  %2 = fptoui %arg1: f64 to i32
// CHECK-NEXT: = llvm.fptoui {{.*}} : f64 to i64
  %3 = fptoui %arg1: f64 to i64
  return
}

// Checking conversion of floating point vectors to integer vector types.
// CHECK-LABEL: @fptoui_vector
func @fptoui_vector(%arg0 : vector<2xf16>, %arg1 : vector<2xf32>, %arg2 : vector<2xf64>) {
// CHECK-NEXT: = llvm.fptoui {{.*}} : vector<2xf16> to vector<2xi32>
  %0 = fptoui %arg0: vector<2xf16> to vector<2xi32>
// CHECK-NEXT: = llvm.fptoui {{.*}} : vector<2xf16> to vector<2xi64>
  %1 = fptoui %arg0: vector<2xf16> to vector<2xi64>
// CHECK-NEXT: = llvm.fptoui {{.*}} : vector<2xf32> to vector<2xi32>
  %2 = fptoui %arg1: vector<2xf32> to vector<2xi32>
// CHECK-NEXT: = llvm.fptoui {{.*}} : vector<2xf32> to vector<2xi64>
  %3 = fptoui %arg1: vector<2xf32> to vector<2xi64>
// CHECK-NEXT: = llvm.fptoui {{.*}} : vector<2xf64> to vector<2xi32>
  %4 = fptoui %arg2: vector<2xf64> to vector<2xi32>
// CHECK-NEXT: = llvm.fptoui {{.*}} : vector<2xf64> to vector<2xi64>
  %5 = fptoui %arg2: vector<2xf64> to vector<2xi64>
  return
}

// Checking conversion of integer vectors to floating point vector types.
// CHECK-LABEL: @uitofp_vector
func @uitofp_vector(%arg0 : vector<2xi16>, %arg1 : vector<2xi32>, %arg2 : vector<2xi64>) {
// CHECK-NEXT: = llvm.uitofp {{.*}} : vector<2xi16> to vector<2xf32>
  %0 = uitofp %arg0: vector<2xi16> to vector<2xf32>
// CHECK-NEXT: = llvm.uitofp {{.*}} : vector<2xi16> to vector<2xf64>
  %1 = uitofp %arg0: vector<2xi16> to vector<2xf64>
// CHECK-NEXT: = llvm.uitofp {{.*}} : vector<2xi32> to vector<2xf32>
  %2 = uitofp %arg1: vector<2xi32> to vector<2xf32>
// CHECK-NEXT: = llvm.uitofp {{.*}} : vector<2xi32> to vector<2xf64>
  %3 = uitofp %arg1: vector<2xi32> to vector<2xf64>
// CHECK-NEXT: = llvm.uitofp {{.*}} : vector<2xi64> to vector<2xf32>
  %4 = uitofp %arg2: vector<2xi64> to vector<2xf32>
// CHECK-NEXT: = llvm.uitofp {{.*}} : vector<2xi64> to vector<2xf64>
  %5 = uitofp %arg2: vector<2xi64> to vector<2xf64>
  return
}

// Checking conversion of integer types to floating point.
// CHECK-LABEL: @fptrunc
func @fptrunc(%arg0 : f32, %arg1 : f64) {
// CHECK-NEXT: = llvm.fptrunc {{.*}} : f32 to f16
  %0 = fptrunc %arg0: f32 to f16
// CHECK-NEXT: = llvm.fptrunc {{.*}} : f64 to f16
  %1 = fptrunc %arg1: f64 to f16
// CHECK-NEXT: = llvm.fptrunc {{.*}} : f64 to f32
  %2 = fptrunc %arg1: f64 to f32
  return
}

// Checking conversion of integer types to floating point.
// CHECK-LABEL: @fptrunc
func @fptrunc_vector(%arg0 : vector<2xf32>, %arg1 : vector<2xf64>) {
// CHECK-NEXT: = llvm.fptrunc {{.*}} : vector<2xf32> to vector<2xf16>
  %0 = fptrunc %arg0: vector<2xf32> to vector<2xf16>
// CHECK-NEXT: = llvm.fptrunc {{.*}} : vector<2xf64> to vector<2xf16>
  %1 = fptrunc %arg1: vector<2xf64> to vector<2xf16>
// CHECK-NEXT: = llvm.fptrunc {{.*}} : vector<2xf64> to vector<2xf32>
  %2 = fptrunc %arg1: vector<2xf64> to vector<2xf32>
  return
}

// Check sign and zero extension and truncation of integers.
// CHECK-LABEL: @integer_extension_and_truncation
func @integer_extension_and_truncation(%arg0 : i3) {
// CHECK-NEXT: = llvm.sext %arg0 : i3 to i6
  %0 = sexti %arg0 : i3 to i6
// CHECK-NEXT: = llvm.zext %arg0 : i3 to i6
  %1 = zexti %arg0 : i3 to i6
// CHECK-NEXT: = llvm.trunc %arg0 : i3 to i2
   %2 = trunci %arg0 : i3 to i2
  return
}

// CHECK-LABEL: @dfs_block_order
func @dfs_block_order(%arg0: i32) -> (i32) {
// CHECK-NEXT:  %[[CST:.*]] = llvm.mlir.constant(42 : i32) : i32
  %0 = constant 42 : i32
// CHECK-NEXT:  llvm.br ^bb2
  br ^bb2

// CHECK-NEXT: ^bb1:
// CHECK-NEXT:  %[[ADD:.*]] = llvm.add %arg0, %[[CST]] : i32
// CHECK-NEXT:  llvm.return %[[ADD]] : i32
^bb1:
  %2 = addi %arg0, %0 : i32
  return %2 : i32

// CHECK-NEXT: ^bb2:
^bb2:
// CHECK-NEXT:  llvm.br ^bb1
  br ^bb1
}

// CHECK-LABEL: func @fcmp(%arg0: f32, %arg1: f32) {
func @fcmp(f32, f32) -> () {
^bb0(%arg0: f32, %arg1: f32):
  // CHECK:      llvm.fcmp "oeq" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "ogt" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "oge" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "olt" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "ole" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "one" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "ord" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "ueq" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "ugt" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "uge" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "ult" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "ule" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "une" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.fcmp "uno" %arg0, %arg1 : f32
  // CHECK-NEXT: llvm.return
  %1 = cmpf oeq, %arg0, %arg1 : f32
  %2 = cmpf ogt, %arg0, %arg1 : f32
  %3 = cmpf oge, %arg0, %arg1 : f32
  %4 = cmpf olt, %arg0, %arg1 : f32
  %5 = cmpf ole, %arg0, %arg1 : f32
  %6 = cmpf one, %arg0, %arg1 : f32
  %7 = cmpf ord, %arg0, %arg1 : f32
  %8 = cmpf ueq, %arg0, %arg1 : f32
  %9 = cmpf ugt, %arg0, %arg1 : f32
  %10 = cmpf uge, %arg0, %arg1 : f32
  %11 = cmpf ult, %arg0, %arg1 : f32
  %12 = cmpf ule, %arg0, %arg1 : f32
  %13 = cmpf une, %arg0, %arg1 : f32
  %14 = cmpf uno, %arg0, %arg1 : f32

  return
}

// CHECK-LABEL: @splat
// CHECK-SAME: %[[A:arg[0-9]+]]: vector<4xf32>
// CHECK-SAME: %[[ELT:arg[0-9]+]]: f32
func @splat(%a: vector<4xf32>, %b: f32) -> vector<4xf32> {
  %vb = splat %b : vector<4xf32>
  %r = mulf %a, %vb : vector<4xf32>
  return %r : vector<4xf32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.undef : vector<4xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<4xf32>
// CHECK-NEXT: %[[SPLAT:[0-9]+]] = llvm.shufflevector %[[V]], %[[UNDEF]] [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK-NEXT: %[[SCALE:[0-9]+]] = llvm.fmul %[[A]], %[[SPLAT]] : vector<4xf32>
// CHECK-NEXT: llvm.return %[[SCALE]] : vector<4xf32>

// CHECK-LABEL: func @view(
// CHECK: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64
func @view(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() : memref<2048xi8>

  // Test two dynamic sizes.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]][%[[ARG2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG0]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %1 = memref.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>

  // Test one dynamic size.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_2:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_2:.*]] = llvm.getelementptr %[[BASE_PTR_2]][%[[ARG2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR_2:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR_2]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR_2]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_2]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %3 = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>

  // Test static sizes.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_3:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_3:.*]] = llvm.getelementptr %[[BASE_PTR_3]][%[[ARG2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR_3:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR_3]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR_3]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_3:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_3]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %5 = memref.view %0[%arg2][] : memref<2048xi8> to memref<64x4xf32>

  // Test view memory space.
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i8, 4>, ptr<i8, 4>, i64, array<1 x i64>, array<1 x i64>)>
  %6 = memref.alloc() : memref<2048xi8, 4>

  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_4:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8, 4>, ptr<i8, 4>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_4:.*]] = llvm.getelementptr %[[BASE_PTR_4]][%[[ARG2]]] : (!llvm.ptr<i8, 4>, i64) -> !llvm.ptr<i8, 4>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR_4:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR_4]] : !llvm.ptr<i8, 4> to !llvm.ptr<f32, 4>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR_4]], %{{.*}}[1] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_4:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_4]], %{{.*}}[2] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  %7 = memref.view %6[%arg2][] : memref<2048xi8, 4> to memref<64x4xf32, 4>

  return
}

// CHECK-LABEL: func @subview(
// CHECK-COUNT-2: !llvm.ptr<f32>,
// CHECK-COUNT-5: {{%[a-zA-Z0-9]*}}: i64,
// CHECK:         %[[ARG0:[a-zA-Z0-9]*]]: i64,
// CHECK:         %[[ARG1:[a-zA-Z0-9]*]]: i64,
// CHECK:         %[[ARG2:.*]]: i64)
// CHECK32-LABEL: func @subview(
// CHECK32-COUNT-2: !llvm.ptr<f32>,
// CHECK32-COUNT-5: {{%[a-zA-Z0-9]*}}: i32,
// CHECK32:         %[[ARG0:[a-zA-Z0-9]*]]: i32,
// CHECK32:         %[[ARG1:[a-zA-Z0-9]*]]: i32,
// CHECK32:         %[[ARG2:.*]]: i32)
func @subview(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]
  // CHECK32: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
  to memref<?x?xf32, offset: ?, strides: [?, ?]>
  return
}

// CHECK-LABEL: func @subview_non_zero_addrspace(
// CHECK-COUNT-2: !llvm.ptr<f32, 3>,
// CHECK-COUNT-5: {{%[a-zA-Z0-9]*}}: i64,
// CHECK:         %[[ARG0:[a-zA-Z0-9]*]]: i64,
// CHECK:         %[[ARG1:[a-zA-Z0-9]*]]: i64,
// CHECK:         %[[ARG2:.*]]: i64)
// CHECK32-LABEL: func @subview_non_zero_addrspace(
// CHECK32-COUNT-2: !llvm.ptr<f32, 3>,
// CHECK32-COUNT-5: {{%[a-zA-Z0-9]*}}: i32,
// CHECK32:         %[[ARG0:[a-zA-Z0-9]*]]: i32,
// CHECK32:         %[[ARG1:[a-zA-Z0-9]*]]: i32,
// CHECK32:         %[[ARG2:.*]]: i32)
func @subview_non_zero_addrspace(%0 : memref<64x4xf32, offset: 0, strides: [4, 1], 3>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]
  // CHECK32: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1], 3>
    to memref<?x?xf32, offset: ?, strides: [?, ?], 3>
  return
}

// CHECK-LABEL: func @subview_const_size(
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG7:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG8:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG9:[a-zA-Z0-9]*]]: i64
// CHECK32-LABEL: func @subview_const_size(
// CHECK32-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK32-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK32-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG7:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG8:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG9:[a-zA-Z0-9]*]]: i32
func @subview_const_size(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]
  // CHECK32: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG7]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG8]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG8]], %[[STRIDE1]] : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[CST2]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG7]], %[[STRIDE0]] : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[CST4]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG7]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG8]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK32: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG8]], %[[STRIDE1]] : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[CST2]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG7]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[CST4]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[%arg0, %arg1][4, 2][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<4x2xf32, offset: ?, strides: [?, ?]>
  return
}

// CHECK-LABEL: func @subview_const_stride(
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG7:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG8:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG9:[a-zA-Z0-9]*]]: i64
// CHECK32-LABEL: func @subview_const_stride(
// CHECK32-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK32-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<f32>,
// CHECK32-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG7:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG8:[a-zA-Z0-9]*]]: i32
// CHECK32-SAME:         %[[ARG9:[a-zA-Z0-9]*]]: i32
func @subview_const_stride(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]
  // CHECK32: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG7]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG8]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG8]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[CST2]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG7]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[CST4]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG7]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG8]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG8]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[CST2]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[ARG7]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[CST4]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][1, 2] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<?x?xf32, offset: ?, strides: [4, 2]>
  return
}

// CHECK-LABEL: func @subview_const_stride_and_offset(
// CHECK32-LABEL: func @subview_const_stride_and_offset(
func @subview_const_stride_and_offset(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>) {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]
  // CHECK32: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]

  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST8:.*]] = llvm.mlir.constant(8 : index)
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[CST8]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST3:.*]] = llvm.mlir.constant(3 : i64)
  // CHECK32: %[[CST1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[CST3]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[CST1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST62:.*]] = llvm.mlir.constant(62 : i64)
  // CHECK32: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[CST62]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[CST4]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[0, 8][62, 3][1, 1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<62x3xf32, offset: 8, strides: [4, 1]>
  return
}

// CHECK-LABEL: func @subview_mixed_static_dynamic(
// CHECK-COUNT-2: !llvm.ptr<f32>,
// CHECK-COUNT-5: {{%[a-zA-Z0-9]*}}: i64,
// CHECK:         %[[ARG0:[a-zA-Z0-9]*]]: i64,
// CHECK:         %[[ARG1:[a-zA-Z0-9]*]]: i64,
// CHECK:         %[[ARG2:.*]]: i64)
// CHECK32-LABEL: func @subview_mixed_static_dynamic(
// CHECK32-COUNT-2: !llvm.ptr<f32>,
// CHECK32-COUNT-5: {{%[a-zA-Z0-9]*}}: i32,
// CHECK32:         %[[ARG0:[a-zA-Z0-9]*]]: i32,
// CHECK32:         %[[ARG1:[a-zA-Z0-9]*]]: i32,
// CHECK32:         %[[ARG2:.*]]: i32)
func @subview_mixed_static_dynamic(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]
  // CHECK32: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1]

  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFM1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFFA1:.*]] = llvm.add %[[OFF]], %[[OFFM1]] : i32
  // CHECK32: %[[CST8:.*]] = llvm.mlir.constant(8 : i64) : i32
  // CHECK32: %[[OFFM2:.*]] = llvm.mul %[[CST8]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFFA2:.*]] = llvm.add %[[OFFA1]], %[[OFFM2]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFFA2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>

  // CHECK32: %[[CST1:.*]] = llvm.mlir.constant(1 : i64) : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG2]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[CST1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST62:.*]] = llvm.mlir.constant(62 : i64) : i32
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[CST62]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[%arg1, 8][62, %arg2][%arg0, 1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<62x?xf32, offset: ?, strides: [?, 1]>
  return
}

// CHECK-LABEL: func @subview_leading_operands(
func @subview_leading_operands(%0 : memref<5x3xf32>, %1: memref<5x?xf32>) {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Alloc ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Aligned ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Offset
  // CHECK: %[[C6:.*]] = llvm.mlir.constant(6 : index) : i64
  // CHECK: llvm.insertvalue %[[C6:.*]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 1: both static.
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.insertvalue %[[C3]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[C1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 0: both static extracted from type.
  // CHECK: %[[C3_2:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[C3_3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: llvm.insertvalue %[[C3_2]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[C3_3]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %2 = memref.subview %0[2][3][1]: memref<5x3xf32> to memref<3x3xf32, offset: 6, strides: [3, 1]>

  return
}

// CHECK-LABEL: func @subview_leading_operands_dynamic(
func @subview_leading_operands_dynamic(%0 : memref<5x?xf32>) {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Alloc ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Aligned ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Extract strides
  // CHECK: %[[ST0:.*]] = llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[ST1:.*]] = llvm.extractvalue %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Compute and insert offset from 2 + dynamic value.
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[C2]], %[[ST0]] : i64
  // CHECK: %[[NEW_OFF:.*]] = llvm.add %[[OFF]], %[[MUL]]  : i64
  // CHECK: llvm.insertvalue %[[NEW_OFF]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 1: static stride 1, dynamic size unchanged from source memref.
  // CHECK: %[[SZ1:.*]] = llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.insertvalue %[[SZ1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[C1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 0: both static.
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[C1_2:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[C1_2]], %[[ST0]]  : i64
  // CHECK: llvm.insertvalue %[[C3]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[MUL]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %1 = memref.subview %0[2][3][1]: memref<5x?xf32> to memref<3x?xf32, offset: ?, strides: [?, 1]>
  return
}

// CHECK-LABEL: func @subview_rank_reducing_leading_operands(
func @subview_rank_reducing_leading_operands(%0 : memref<5x3xf32>) {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Alloc ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Aligned ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Extract strides
  // CHECK: %[[ST0:.*]] = llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[ST1:.*]] = llvm.extractvalue %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Offset
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: llvm.insertvalue %[[C3:.*]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Sizes and strides @rank 0: both static.
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.insertvalue %[[C3]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.subview %0[1][1][1]: memref<5x3xf32> to memref<3xf32, offset: 3, strides: [1]>

  return
}


// -----

// CHECK-LABEL: func @atomic_rmw
func @atomic_rmw(%I : memref<10xi32>, %ival : i32, %F : memref<10xf32>, %fval : f32, %i : index) {
  atomic_rmw assign %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw xchg %{{.*}}, %{{.*}} acq_rel
  atomic_rmw addi %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw add %{{.*}}, %{{.*}} acq_rel
  atomic_rmw maxs %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw max %{{.*}}, %{{.*}} acq_rel
  atomic_rmw mins %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw min %{{.*}}, %{{.*}} acq_rel
  atomic_rmw maxu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umax %{{.*}}, %{{.*}} acq_rel
  atomic_rmw minu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umin %{{.*}}, %{{.*}} acq_rel
  atomic_rmw addf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} acq_rel
  return
}

// -----

// CHECK-LABEL: func @generic_atomic_rmw
func @generic_atomic_rmw(%I : memref<10xi32>, %i : index) -> i32 {
  %x = generic_atomic_rmw %I[%i] : memref<10xi32> {
    ^bb0(%old_value : i32):
      %c1 = constant 1 : i32
      atomic_yield %c1 : i32
  }
  // CHECK: [[init:%.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
  // CHECK-NEXT: llvm.br ^bb1([[init]] : i32)
  // CHECK-NEXT: ^bb1([[loaded:%.*]]: i32):
  // CHECK-NEXT: [[c1:%.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-NEXT: [[pair:%.*]] = llvm.cmpxchg %{{.*}}, [[loaded]], [[c1]]
  // CHECK-SAME:                    acq_rel monotonic : i32
  // CHECK-NEXT: [[new:%.*]] = llvm.extractvalue [[pair]][0]
  // CHECK-NEXT: [[ok:%.*]] = llvm.extractvalue [[pair]][1]
  // CHECK-NEXT: llvm.cond_br [[ok]], ^bb2, ^bb1([[new]] : i32)
  // CHECK-NEXT: ^bb2:
  %c2 = constant 2 : i32
  %add = addi %c2, %x : i32
  return %add : i32
  // CHECK-NEXT: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
  // CHECK-NEXT: [[add:%.*]] = llvm.add [[c2]], [[new]] : i32
  // CHECK-NEXT: llvm.return [[add]]
}

// -----

// CHECK-LABEL: func @assume_alignment
func @assume_alignment(%0 : memref<4x4xf16>) {
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[MEMREF:.*]][1] : !llvm.struct<(ptr<f16>, ptr<f16>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK-NEXT: %[[MASK:.*]] = llvm.mlir.constant(15 : index) : i64
  // CHECK-NEXT: %[[INT:.*]] = llvm.ptrtoint %[[PTR]] : !llvm.ptr<f16> to i64
  // CHECK-NEXT: %[[MASKED_PTR:.*]] = llvm.and %[[INT]], %[[MASK:.*]] : i64
  // CHECK-NEXT: %[[CONDITION:.*]] = llvm.icmp "eq" %[[MASKED_PTR]], %[[ZERO]] : i64
  // CHECK-NEXT: "llvm.intr.assume"(%[[CONDITION]]) : (i1) -> ()
  memref.assume_alignment %0, 16 : memref<4x4xf16>
  return
}

// -----

// CHECK-LABEL: func @memref_index
// CHECK-SAME: %arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>,
// CHECK-SAME: %arg2: i64, %arg3: i64, %arg4: i64)
// CHECK-SAME: -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK32-LABEL: func @memref_index
// CHECK32-SAME: %arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>,
// CHECK32-SAME: %arg2: i32, %arg3: i32, %arg4: i32)
// CHECK32-SAME: -> !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>
func @memref_index(%arg0: memref<32xindex>) -> memref<32xindex> {
  return %arg0 : memref<32xindex>
}

// -----

// CHECK-LABEL: func @rank_of_unranked
// CHECK32-LABEL: func @rank_of_unranked
func @rank_of_unranked(%unranked: memref<*xi32>) {
  %rank = rank %unranked : memref<*xi32>
  return
}
// CHECK-NEXT: llvm.mlir.undef
// CHECK-NEXT: llvm.insertvalue
// CHECK-NEXT: llvm.insertvalue
// CHECK-NEXT: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK32: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, ptr<i8>)>

// CHECK-LABEL: func @rank_of_ranked
// CHECK32-LABEL: func @rank_of_ranked
func @rank_of_ranked(%ranked: memref<?xi32>) {
  %rank = rank %ranked : memref<?xi32>
  return
}
// CHECK: llvm.mlir.constant(1 : index) : i64
// CHECK32: llvm.mlir.constant(1 : index) : i32

// -----

// CHECK-LABEL: func @dim_of_unranked
// CHECK32-LABEL: func @dim_of_unranked
func @dim_of_unranked(%unranked: memref<*xi32>) -> index {
  %c0 = constant 0 : index
  %dim = memref.dim %unranked, %c0 : memref<*xi32>
  return %dim : index
}
// CHECK-NEXT: llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK-NEXT: llvm.insertvalue
// CHECK-NEXT: %[[UNRANKED_DESC:.*]] = llvm.insertvalue
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64

// CHECK-NEXT: %[[RANKED_DESC:.*]] = llvm.extractvalue %[[UNRANKED_DESC]][1]
// CHECK-SAME:   : !llvm.struct<(i64, ptr<i8>)>

// CHECK-NEXT: %[[ZERO_D_DESC:.*]] = llvm.bitcast %[[RANKED_DESC]]
// CHECK-SAME:   : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64)>>

// CHECK-NEXT: %[[C2_i32:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[C0_:.*]] = llvm.mlir.constant(0 : index) : i64

// CHECK-NEXT: %[[OFFSET_PTR:.*]] = llvm.getelementptr %[[ZERO_D_DESC]]{{\[}}
// CHECK-SAME:   %[[C0_]], %[[C2_i32]]] : (!llvm.ptr<struct<(ptr<i32>, ptr<i32>,
// CHECK-SAME:   i64)>>, i64, i32) -> !llvm.ptr<i64>

// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[INDEX_INC:.*]] = llvm.add %[[C1]], %[[C0]] : i64

// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[OFFSET_PTR]]{{\[}}
// CHECK-SAME:   %[[INDEX_INC]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>

// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr<i64>
// CHECK-NEXT: llvm.return %[[SIZE]] : i64

// CHECK32: %[[SIZE:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
// CHECK32-NEXT: llvm.return %[[SIZE]] : i32
