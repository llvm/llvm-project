// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-math-to-llvm,convert-arith-to-llvm),convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" %s | FileCheck %s

// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-math-to-llvm,convert-arith-to-llvm{index-bitwidth=32}),convert-func-to-llvm{index-bitwidth=32},convert-cf-to-llvm{index-bitwidth=32},reconcile-unrealized-casts)" %s | FileCheck --check-prefix=CHECK32 %s

// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-math-to-llvm,convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)" %s | FileCheck --check-prefix=CHECK-NO-CF %s

// RUN: mlir-opt -transform-interpreter %s | FileCheck --check-prefix=CHECK32 %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm="filter-dialects=arith,cf,func,math" %s | FileCheck %s

// CHECK-LABEL: func @empty() {
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }
func.func @empty() {
^bb0:
  return
}

// CHECK-LABEL: llvm.func @body(i64)
func.func private @body(index)

// CHECK-LABEL: func @simple_loop() {
// CHECK32-LABEL: func @simple_loop() {
func.func @simple_loop() {
^bb0:
// CHECK-NEXT:  llvm.br ^bb1
// CHECK32-NEXT:  llvm.br ^bb1
  cf.br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK32-NEXT: ^bb1:	// pred: ^bb0
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i32
// CHECK32-NEXT:  llvm.br ^bb2({{.*}} : i32)
^bb1:	// pred: ^bb0
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  cf.br ^bb2(%c1 : index)

// CHECK:      ^bb2({{.*}}: i64):	// 2 preds: ^bb1, ^bb3
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
// CHECK32:      ^bb2({{.*}}: i32):	// 2 preds: ^bb1, ^bb3
// CHECK32-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i32
// CHECK32-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %c42 : index
  cf.cond_br %1, ^bb3, ^bb4

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
  %c1_0 = arith.constant 1 : index
  %2 = arith.addi %0, %c1_0 : index
  cf.br ^bb2(%2 : index)

// CHECK:      ^bb4:	// pred: ^bb2
// CHECK-NEXT:  llvm.return
^bb4:	// pred: ^bb2
  return
}

// CHECK-LABEL: func @simple_caller() {
// CHECK-NEXT:  llvm.call @simple_loop() : () -> ()
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }
func.func @simple_caller() {
^bb0:
  call @simple_loop() : () -> ()
  return
}

// Check that function call attributes persist during conversion.
// CHECK-LABEL: @call_with_attributes
func.func @call_with_attributes() {
  // CHECK: llvm.call @simple_loop() {baz = [1, 2, 3, 4], foo = "bar"} : () -> ()
  call @simple_loop() {foo="bar", baz=[1,2,3,4]} : () -> ()
  return
}

// CHECK-LABEL: func @ml_caller() {
// CHECK-NEXT:  llvm.call @simple_loop() : () -> ()
// CHECK-NEXT:  llvm.call @more_imperfectly_nested_loops() : () -> ()
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }
func.func @ml_caller() {
^bb0:
  call @simple_loop() : () -> ()
  call @more_imperfectly_nested_loops() : () -> ()
  return
}

// CHECK-LABEL: llvm.func @body_args(i64) -> i64
// CHECK32-LABEL: llvm.func @body_args(i32) -> i32
// CHECK-NO-CF-LABEL: llvm.func @body_args(i64) -> i64
func.func private @body_args(index) -> index
// CHECK-LABEL: llvm.func @other(i64, i32) -> i32
// CHECK32-LABEL: llvm.func @other(i32, i32) -> i32
func.func private @other(index, i32) -> i32

// CHECK-LABEL: func @func_args(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  llvm.br ^bb1
// CHECK32-LABEL: func @func_args(%arg0: i32, %arg1: i32) -> i32 {
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(0 : i32) : i32
// CHECK32-NEXT:  llvm.br ^bb1
func.func @func_args(i32, i32) -> i32 {
^bb0(%arg0: i32, %arg1: i32):
  %c0_i32 = arith.constant 0 : i32
  cf.br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
// CHECK32-NEXT: ^bb1:	// pred: ^bb0
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i32
// CHECK32-NEXT:  llvm.br ^bb2({{.*}} : i32)
^bb1:	// pred: ^bb0
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  cf.br ^bb2(%c0 : index)

// CHECK-NEXT: ^bb2({{.*}}: i64):	// 2 preds: ^bb1, ^bb3
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
// CHECK32-NEXT: ^bb2({{.*}}: i32):	// 2 preds: ^bb1, ^bb3
// CHECK32-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i32
// CHECK32-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb4
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %c42 : index
  cf.cond_br %1, ^bb3, ^bb4

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
  %c1 = arith.constant 1 : index
  %6 = arith.addi %0, %c1 : index
  cf.br ^bb2(%6 : index)

// CHECK-NEXT: ^bb4:	// pred: ^bb2
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.call @other({{.*}}, {{.*}}) : (i64, i32) -> i32
// CHECK-NEXT:  llvm.return {{.*}} : i32
// CHECK32-NEXT: ^bb4:	// pred: ^bb2
// CHECK32-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i32
// CHECK32-NEXT:  {{.*}} = llvm.call @other({{.*}}, {{.*}}) : (i32, i32) -> i32
// CHECK32-NEXT:  llvm.return {{.*}} : i32
^bb4:	// pred: ^bb2
  %c0_0 = arith.constant 0 : index
  %7 = call @other(%c0_0, %c0_i32) : (index, i32) -> i32
  return %7 : i32
}

// CHECK-LABEL: llvm.func @pre(i64)
// CHECK32-LABEL: llvm.func @pre(i32)
func.func private @pre(index)

// CHECK-LABEL: llvm.func @body2(i64, i64)
// CHECK32-LABEL: llvm.func @body2(i32, i32)
func.func private @body2(index, index)

// CHECK-LABEL: llvm.func @post(i64)
// CHECK32-LABEL: llvm.func @post(i32)
func.func private @post(index)

// CHECK-LABEL: func @imperfectly_nested_loops() {
// CHECK-NEXT:  llvm.br ^bb1
func.func @imperfectly_nested_loops() {
^bb0:
  cf.br ^bb1

// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
^bb1:	// pred: ^bb0
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  cf.br ^bb2(%c0 : index)

// CHECK-NEXT: ^bb2({{.*}}: i64):	// 2 preds: ^bb1, ^bb7
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb3, ^bb8
^bb2(%0: index):	// 2 preds: ^bb1, ^bb7
  %1 = arith.cmpi slt, %0, %c42 : index
  cf.cond_br %1, ^bb3, ^bb8

// CHECK-NEXT: ^bb3:
// CHECK-NEXT:  llvm.call @pre({{.*}}) : (i64) -> ()
// CHECK-NEXT:  llvm.br ^bb4
^bb3:	// pred: ^bb2
  call @pre(%0) : (index) -> ()
  cf.br ^bb4

// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(7 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(56 : index) : i64
// CHECK-NEXT:  llvm.br ^bb5({{.*}} : i64)
^bb4:	// pred: ^bb3
  %c7 = arith.constant 7 : index
  %c56 = arith.constant 56 : index
  cf.br ^bb5(%c7 : index)

// CHECK-NEXT: ^bb5({{.*}}: i64):	// 2 preds: ^bb4, ^bb6
// CHECK-NEXT:  {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.cond_br {{.*}}, ^bb6, ^bb7
^bb5(%2: index):	// 2 preds: ^bb4, ^bb6
  %3 = arith.cmpi slt, %2, %c56 : index
  cf.cond_br %3, ^bb6, ^bb7

// CHECK-NEXT: ^bb6:	// pred: ^bb5
// CHECK-NEXT:  llvm.call @body2({{.*}}, {{.*}}) : (i64, i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb5({{.*}} : i64)
^bb6:	// pred: ^bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = arith.constant 2 : index
  %4 = arith.addi %2, %c2 : index
  cf.br ^bb5(%4 : index)

// CHECK-NEXT: ^bb7:	// pred: ^bb5
// CHECK-NEXT:  llvm.call @post({{.*}}) : (i64) -> ()
// CHECK-NEXT:  {{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  {{.*}} = llvm.add {{.*}}, {{.*}} : i64
// CHECK-NEXT:  llvm.br ^bb2({{.*}} : i64)
^bb7:	// pred: ^bb5
  call @post(%0) : (index) -> ()
  %c1 = arith.constant 1 : index
  %5 = arith.addi %0, %c1 : index
  cf.br ^bb2(%5 : index)

// CHECK-NEXT: ^bb8:	// pred: ^bb2
// CHECK-NEXT:  llvm.return
^bb8:	// pred: ^bb2
  return
}

// CHECK-LABEL: llvm.func @mid(i64)
func.func private @mid(index)

// CHECK-LABEL: llvm.func @body3(i64, i64)
func.func private @body3(index, index)

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
func.func @more_imperfectly_nested_loops() {
^bb0:
  cf.br ^bb1
^bb1:	// pred: ^bb0
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  cf.br ^bb2(%c0 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb11
  %1 = arith.cmpi slt, %0, %c42 : index
  cf.cond_br %1, ^bb3, ^bb12
^bb3:	// pred: ^bb2
  call @pre(%0) : (index) -> ()
  cf.br ^bb4
^bb4:	// pred: ^bb3
  %c7 = arith.constant 7 : index
  %c56 = arith.constant 56 : index
  cf.br ^bb5(%c7 : index)
^bb5(%2: index):	// 2 preds: ^bb4, ^bb6
  %3 = arith.cmpi slt, %2, %c56 : index
  cf.cond_br %3, ^bb6, ^bb7
^bb6:	// pred: ^bb5
  call @body2(%0, %2) : (index, index) -> ()
  %c2 = arith.constant 2 : index
  %4 = arith.addi %2, %c2 : index
  cf.br ^bb5(%4 : index)
^bb7:	// pred: ^bb5
  call @mid(%0) : (index) -> ()
  cf.br ^bb8
^bb8:	// pred: ^bb7
  %c18 = arith.constant 18 : index
  %c37 = arith.constant 37 : index
  cf.br ^bb9(%c18 : index)
^bb9(%5: index):	// 2 preds: ^bb8, ^bb10
  %6 = arith.cmpi slt, %5, %c37 : index
  cf.cond_br %6, ^bb10, ^bb11
^bb10:	// pred: ^bb9
  call @body3(%0, %5) : (index, index) -> ()
  %c3 = arith.constant 3 : index
  %7 = arith.addi %5, %c3 : index
  cf.br ^bb9(%7 : index)
^bb11:	// pred: ^bb9
  call @post(%0) : (index) -> ()
  %c1 = arith.constant 1 : index
  %8 = arith.addi %0, %c1 : index
  cf.br ^bb2(%8 : index)
^bb12:	// pred: ^bb2
  return
}

// CHECK-LABEL: llvm.func @get_i64() -> i64
func.func private @get_i64() -> (i64)
// CHECK-LABEL: llvm.func @get_f32() -> f32
func.func private @get_f32() -> (f32)
// CHECK-LABEL: llvm.func @get_c16() -> !llvm.struct<(f16, f16)>
func.func private @get_c16() -> (complex<f16>)
// CHECK-LABEL: llvm.func @get_c32() -> !llvm.struct<(f32, f32)>
func.func private @get_c32() -> (complex<f32>)
// CHECK-LABEL: llvm.func @get_c64() -> !llvm.struct<(f64, f64)>
func.func private @get_c64() -> (complex<f64>)
// CHECK-LABEL: llvm.func @get_memref() -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK32-LABEL: llvm.func @get_memref() -> !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
func.func private @get_memref() -> (memref<42x?x10x?xf32>)

// CHECK-LABEL: llvm.func @multireturn() -> !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)> {
// CHECK32-LABEL: llvm.func @multireturn() -> !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)> {
func.func @multireturn() -> (i64, f32, memref<42x?x10x?xf32>) {
^bb0:
// CHECK-NEXT:  {{.*}} = llvm.call @get_i64() : () -> i64
// CHECK-NEXT:  {{.*}} = llvm.call @get_f32() : () -> f32
// CHECK-NEXT:  {{.*}} = llvm.call @get_memref() : () -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK32-NEXT:  {{.*}} = llvm.call @get_i64() : () -> i64
// CHECK32-NEXT:  {{.*}} = llvm.call @get_f32() : () -> f32
// CHECK32-NEXT:  {{.*}} = llvm.call @get_memref() : () -> !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
  %0 = call @get_i64() : () -> (i64)
  %1 = call @get_f32() : () -> (f32)
  %2 = call @get_memref() : () -> (memref<42x?x10x?xf32>)
// CHECK-NEXT:  {{.*}} = llvm.mlir.undef : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  llvm.return {{.*}} : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.mlir.undef : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  llvm.return {{.*}} : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
  return %0, %1, %2 : i64, f32, memref<42x?x10x?xf32>
}


// CHECK-LABEL: llvm.func @multireturn_caller() {
// CHECK32-LABEL: llvm.func @multireturn_caller() {
func.func @multireturn_caller() {
^bb0:
// CHECK-NEXT:  {{.*}} = llvm.call @multireturn() : () -> !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.call @multireturn() : () -> !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
// CHECK32-NEXT:  {{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>)>
  %0:3 = call @multireturn() : () -> (i64, f32, memref<42x?x10x?xf32>)
  %1 = arith.constant 42 : i64
// CHECK:       {{.*}} = llvm.add {{.*}}, {{.*}} : i64
  %2 = arith.addi %0#0, %1 : i64
  %3 = arith.constant 42.0 : f32
// CHECK:       {{.*}} = llvm.fadd {{.*}}, {{.*}} : f32
  %4 = arith.addf %0#1, %3 : f32
  %5 = arith.constant 0 : index
  return
}

// CHECK-LABEL: @dfs_block_order
func.func @dfs_block_order(%arg0: i32) -> (i32) {
// CHECK-NEXT:  %[[CST:.*]] = llvm.mlir.constant(42 : i32) : i32
  %0 = arith.constant 42 : i32
// CHECK-NEXT:  llvm.br ^bb2
  cf.br ^bb2

// CHECK-NEXT: ^bb1:
// CHECK-NEXT:  %[[ADD:.*]] = llvm.add %arg0, %[[CST]] : i32
// CHECK-NEXT:  llvm.return %[[ADD]] : i32
^bb1:
  %2 = arith.addi %arg0, %0 : i32
  return %2 : i32

// CHECK-NEXT: ^bb2:
^bb2:
// CHECK-NEXT:  llvm.br ^bb1
  cf.br ^bb1
}

// CHECK-LABEL: func @ceilf(
// CHECK-SAME: f32
func.func @ceilf(%arg0 : f32) {
  // CHECK: llvm.intr.ceil(%arg0) : (f32) -> f32
  %0 = math.ceil %arg0 : f32
  func.return
}

// CHECK-LABEL: func @floorf(
// CHECK-SAME: f32
func.func @floorf(%arg0 : f32) {
  // CHECK: llvm.intr.floor(%arg0) : (f32) -> f32
  %0 = math.floor %arg0 : f32
  func.return
}

// Wrap the following tests in a module to control the place where 
// `llvm.func @abort()` is produced.
module {
// Lowers `cf.assert` to a function call to `abort` if the assertion is violated.
// CHECK: llvm.func @abort()
// CHECK-LABEL: @assert_test_function
// CHECK-SAME:  (%[[ARG:.*]]: i1)
func.func @assert_test_function(%arg : i1) {
  // CHECK: llvm.cond_br %[[ARG]], ^[[CONTINUATION_BLOCK:.*]], ^[[FAILURE_BLOCK:.*]]
  // CHECK: ^[[CONTINUATION_BLOCK]]:
  // CHECK: llvm.return
  // CHECK: ^[[FAILURE_BLOCK]]:
  // CHECK: llvm.call @abort() : () -> ()
  // CHECK: llvm.unreachable
  cf.assert %arg, "Computer says no"
  return
}
}

// This should not trigger an assertion by creating an LLVM::CallOp with a
// nullptr result type.

// CHECK-LABEL: @call_zero_result_func
func.func @call_zero_result_func() {
  // CHECK: call @zero_result_func
  call @zero_result_func() : () -> ()
  return
}
func.func private @zero_result_func()

// CHECK-LABEL: func @fmaf(
// CHECK-SAME: %[[ARG0:.*]]: f32
// CHECK-SAME: %[[ARG1:.*]]: vector<4xf32>
func.func @fmaf(%arg0: f32, %arg1: vector<4xf32>) {
  // CHECK: %[[S:.*]] = llvm.intr.fma(%[[ARG0]], %[[ARG0]], %[[ARG0]]) : (f32, f32, f32) -> f32
  %0 = math.fma %arg0, %arg0, %arg0 : f32
  // CHECK: %[[V:.*]] = llvm.intr.fma(%[[ARG1]], %[[ARG1]], %[[ARG1]]) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %1 = math.fma %arg1, %arg1, %arg1 : vector<4xf32>
  func.return
}

// CHECK-LABEL: func @switchi8(
func.func @switchi8(%arg0 : i8) -> i32 {
  cf.switch %arg0 : i8, [
    default: ^bb1,
    42: ^bb1,
    43: ^bb3
  ]
^bb1:
  %c_1 = arith.constant 1 : i32
  func.return %c_1 : i32
^bb3:
  %c_42 = arith.constant 42 : i32
  func.return %c_42: i32
}
// CHECK:     llvm.switch %arg0 : i8, ^bb1 [
// CHECK-NEXT:       42: ^bb1,
// CHECK-NEXT:       43: ^bb2
// CHECK-NEXT:     ]
// CHECK:   ^bb1:  // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:     %[[E0:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     llvm.return %[[E0]] : i32
// CHECK:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     %[[E1:.+]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT:     llvm.return %[[E1]] : i32
// CHECK-NEXT:   }

// Convert the entry block but not the unstructured control flow.

// CHECK-NO-CF-LABEL: llvm.func @index_arg(
//  CHECK-NO-CF-SAME:     %[[arg0:.*]]: i64) -> i64 {
//       CHECK-NO-CF:   %[[cast:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : i64 to index
//       CHECK-NO-CF:   cf.br ^[[bb1:.*]](%[[cast]] : index)
//       CHECK-NO-CF: ^[[bb1]](%[[arg1:.*]]: index):
//       CHECK-NO-CF:   %[[cast2:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i64
//       CHECK-NO-CF:   llvm.return %[[cast2]] : i64
func.func @index_arg(%arg0: index) -> index {
  cf.br ^bb1(%arg0 : index)
^bb1(%arg1: index):
  return %arg1 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%toplevel_module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %toplevel_module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.dialect_to_llvm "math"
      transform.apply_conversion_patterns.dialect_to_llvm "arith"
      transform.apply_conversion_patterns.dialect_to_llvm "cf"
      transform.apply_conversion_patterns.func.func_to_llvm
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
        {index_bitwidth = 32, use_opaque_pointers = true}
    } {
      legal_dialects = ["llvm"], 
      partial_conversion
    } : !transform.any_op
    transform.yield
  }
}
