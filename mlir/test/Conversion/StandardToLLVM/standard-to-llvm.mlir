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
// CHECK:  = llvm.fsub %arg0, %arg1 : f32
  %0 = subf %arg0, %arg1: f32
// CHECK: = llvm.sub %arg2, %arg3 : i32
  %1 = subi %arg2, %arg3: i32
// CHECK: = llvm.icmp "slt" %arg2, %1 : i32
  %2 = cmpi slt, %arg2, %1 : i32
// CHECK: = llvm.sdiv %arg2, %arg3 : i32
  %3 = divi_signed %arg2, %arg3 : i32
// CHECK: = llvm.udiv %arg2, %arg3 : i32
  %4 = divi_unsigned %arg2, %arg3 : i32
// CHECK: = llvm.srem %arg2, %arg3 : i32
  %5 = remi_signed %arg2, %arg3 : i32
// CHECK: = llvm.urem %arg2, %arg3 : i32
  %6 = remi_unsigned %arg2, %arg3 : i32
// CHECK: = llvm.select %2, %arg2, %arg3 : i1, i32
  %7 = select %2, %arg2, %arg3 : i32
// CHECK: = llvm.fdiv %arg0, %arg1 : f32
  %8 = divf %arg0, %arg1 : f32
// CHECK: = llvm.frem %arg0, %arg1 : f32
  %9 = remf %arg0, %arg1 : f32
// CHECK: = llvm.and %arg2, %arg3 : i32
  %10 = and %arg2, %arg3 : i32
// CHECK: = llvm.or %arg2, %arg3 : i32
  %11 = or %arg2, %arg3 : i32
// CHECK: = llvm.xor %arg2, %arg3 : i32
  %12 = xor %arg2, %arg3 : i32
// CHECK: = llvm.mlir.constant(7.900000e-01 : f64) : f64
  %15 = constant 7.9e-01 : f64
// CHECK: = llvm.shl %arg2, %arg3 : i32
  %16 = shift_left %arg2, %arg3 : i32
// CHECK: = llvm.ashr %arg2, %arg3 : i32
  %17 = shift_right_signed %arg2, %arg3 : i32
// CHECK: = llvm.lshr %arg2, %arg3 : i32
  %18 = shift_right_unsigned %arg2, %arg3 : i32
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

// CHECK-LABEL: func @ceilf(
// CHECK-SAME: f32
func @ceilf(%arg0 : f32) {
  // CHECK: "llvm.intr.ceil"(%arg0) : (f32) -> f32
  %0 = ceilf %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @floorf(
// CHECK-SAME: f32
func @floorf(%arg0 : f32) {
  // CHECK: "llvm.intr.floor"(%arg0) : (f32) -> f32
  %0 = floorf %arg0 : f32
  std.return
}

// -----

// Lowers `assert` to a function call to `abort` if the assertion is violated.
// CHECK: llvm.func @abort()
// CHECK-LABEL: @assert_test_function
// CHECK-SAME:  (%[[ARG:.*]]: i1)
func @assert_test_function(%arg : i1) {
  // CHECK: llvm.cond_br %[[ARG]], ^[[CONTINUATION_BLOCK:.*]], ^[[FAILURE_BLOCK:.*]]
  // CHECK: ^[[CONTINUATION_BLOCK]]:
  // CHECK: llvm.return
  // CHECK: ^[[FAILURE_BLOCK]]:
  // CHECK: llvm.call @abort() : () -> ()
  // CHECK: llvm.unreachable
  assert %arg, "Computer says no"
  return
}

// -----

// This should not trigger an assertion by creating an LLVM::CallOp with a
// nullptr result type.

// CHECK-LABEL: @call_zero_result_func
func @call_zero_result_func() {
  // CHECK: call @zero_result_func
  call @zero_result_func() : () -> ()
  return
}
func private @zero_result_func()

// -----

// CHECK-LABEL: func @fmaf(
// CHECK-SAME: %[[ARG0:.*]]: f32
// CHECK-SAME: %[[ARG1:.*]]: vector<4xf32>
func @fmaf(%arg0: f32, %arg1: vector<4xf32>) {
  // CHECK: %[[S:.*]] = "llvm.intr.fma"(%[[ARG0]], %[[ARG0]], %[[ARG0]]) : (f32, f32, f32) -> f32
  %0 = fmaf %arg0, %arg0, %arg0 : f32
  // CHECK: %[[V:.*]] = "llvm.intr.fma"(%[[ARG1]], %[[ARG1]], %[[ARG1]]) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %1 = fmaf %arg1, %arg1, %arg1 : vector<4xf32>
  std.return
}

// -----

// CHECK-LABEL: func @index_vector(
// CHECK-SAME: %[[ARG0:.*]]: vector<4xi64>
func @index_vector(%arg0: vector<4xindex>) {
  // CHECK: %[[CST:.*]] = llvm.mlir.constant(dense<[0, 1, 2, 3]> : vector<4xindex>) : vector<4xi64>
  %0 = constant dense<[0, 1, 2, 3]> : vector<4xindex>
  // CHECK: %[[V:.*]] = llvm.add %[[ARG0]], %[[CST]] : vector<4xi64>
  %1 = addi %arg0, %0 : vector<4xindex>
  std.return
}

// -----

// CHECK-LABEL: @bitcast_1d
func @bitcast_1d(%arg0: vector<2xf32>) {
  // CHECK: llvm.bitcast %{{.*}} : vector<2xf32> to vector<2xi32>
  std.bitcast %arg0 : vector<2xf32> to vector<2xi32>
  return
}

// -----

// CHECK-LABEL: func @cmpf_2dvector(
func @cmpf_2dvector(%arg0 : vector<4x3xf32>, %arg1 : vector<4x3xf32>) {
  // CHECK: %[[EXTRACT1:.*]] = llvm.extractvalue %arg0[0] : !llvm.array<4 x vector<3xf32>>
  // CHECK: %[[EXTRACT2:.*]] = llvm.extractvalue %arg1[0] : !llvm.array<4 x vector<3xf32>>
  // CHECK: %[[CMP:.*]] = llvm.fcmp "olt" %[[EXTRACT1]], %[[EXTRACT2]] : vector<3xf32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[CMP]], %0[0] : !llvm.array<4 x vector<3xi1>>
  %0 = cmpf olt, %arg0, %arg1 : vector<4x3xf32>
  std.return
}

// -----

// CHECK-LABEL: func @cmpi_2dvector(
func @cmpi_2dvector(%arg0 : vector<4x3xi32>, %arg1 : vector<4x3xi32>) {
  // CHECK: %[[EXTRACT1:.*]] = llvm.extractvalue %arg0[0] : !llvm.array<4 x vector<3xi32>>
  // CHECK: %[[EXTRACT2:.*]] = llvm.extractvalue %arg1[0] : !llvm.array<4 x vector<3xi32>>
  // CHECK: %[[CMP:.*]] = llvm.icmp "ult" %[[EXTRACT1]], %[[EXTRACT2]] : vector<3xi32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[CMP]], %0[0] : !llvm.array<4 x vector<3xi1>>
  %0 = cmpi ult, %arg0, %arg1 : vector<4x3xi32>
  std.return
}

// -----

// CHECK-LABEL: func @select_2dvector(
func @select_2dvector(%arg0 : vector<4x3xi1>, %arg1 : vector<4x3xi32>, %arg2 : vector<4x3xi32>) {
  // CHECK: %[[EXTRACT1:.*]] = llvm.extractvalue %arg0[0] : !llvm.array<4 x vector<3xi1>>
  // CHECK: %[[EXTRACT2:.*]] = llvm.extractvalue %arg1[0] : !llvm.array<4 x vector<3xi32>>
  // CHECK: %[[EXTRACT3:.*]] = llvm.extractvalue %arg2[0] : !llvm.array<4 x vector<3xi32>>
  // CHECK: %[[SELECT:.*]] = llvm.select %[[EXTRACT1]], %[[EXTRACT2]], %[[EXTRACT3]] : vector<3xi1>, vector<3xi32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[SELECT]], %0[0] : !llvm.array<4 x vector<3xi32>>
  %0 = select %arg0, %arg1, %arg2 : vector<4x3xi1>, vector<4x3xi32>
  std.return
}
