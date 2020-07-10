; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare <4 x float> @llvm.matrix.transpose.v4f32(<4 x float>, i32, i32)
define <4 x float> @transpose(<4 x float> %m, i32 %arg) {
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.3 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.2, i32 %arg, i32 2)
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.4 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.3, i32 2, i32 %arg)
  %result.0 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %m, i32 0, i32 0)
  %result.1 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.0, i32 3, i32 2)
  %result.2 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.1, i32 2, i32 1)
  %result.3 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.2, i32 %arg, i32 2)
  %result.4 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.3, i32 2, i32 %arg)
  ret <4 x float> %result.4
}

declare <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float>, <4 x float>, i32, i32, i32)
define <4 x float> @multiply(<4 x float> %m, i32 %arg) {
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.3 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.2, <4 x float> %m, i32 %arg, i32 2, i32 1)
  %result.0 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %m, <4 x float> %m, i32 0, i32 0, i32 0)
  %result.1 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.0, <4 x float> %m, i32 3, i32 2, i32 2)
  %result.2 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.1, <4 x float> %m, i32 2, i32 2, i32 1)
  %result.3 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.2, <4 x float> %m, i32 %arg, i32 2, i32 1)
  ret <4 x float> %result.3
}

declare <4 x float> @llvm.matrix.column.major.load.v4f32.p0v4f32(<4 x float>*, i64, i1, i32, i32)
declare <6 x float> @llvm.matrix.column.major.load.v6f32.p0v6f32(<6 x float>*, i64, i1, i32, i32)
define <4 x float> @column.major_load(<4 x float>* %m, <6 x float>* %n, i32 %arg) {
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.3 = call <6 x float> @llvm.matrix.column.major.load.v6f32.p0v6f32(<6 x float>* %n, i64 2, i1 true, i32 3, i32 %arg)
  %result.0 = call <4 x float> @llvm.matrix.column.major.load.v4f32.p0v4f32(<4 x float>* %m, i64 0, i1 false, i32 0, i32 0)
  %result.1 = call <4 x float> @llvm.matrix.column.major.load.v4f32.p0v4f32(<4 x float>* %m, i64 2, i1 false, i32 1, i32 2)
  %result.2 = call <6 x float> @llvm.matrix.column.major.load.v6f32.p0v6f32(<6 x float>* %n, i64 2, i1 true, i32 3, i32 3)
  %result.3 = call <6 x float> @llvm.matrix.column.major.load.v6f32.p0v6f32(<6 x float>* %n, i64 2, i1 true, i32 3, i32 %arg)
  ret <4 x float> %result.1
}

declare void @llvm.matrix.column.major.store.v4f32.p0v4f32(<4 x float>, <4 x float>*, i64, i1, i32, i32)
declare void @llvm.matrix.column.major.store.v6f32.p0v6f32(<6 x float>, <6 x float>*, i64, i1, i32, i32)
define void @column.major_store(<4 x float>* %m, <6 x float>* %n, i64 %arg) {
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
  call void @llvm.matrix.column.major.store.v4f32.p0v4f32(<4 x float> zeroinitializer, <4 x float>* %m, i64 0, i1 false, i32 0, i32 0)
  call void @llvm.matrix.column.major.store.v4f32.p0v4f32(<4 x float> zeroinitializer, <4 x float>* %m, i64 2, i1 false, i32 1, i32 2)
  call void @llvm.matrix.column.major.store.v6f32.p0v6f32(<6 x float> zeroinitializer, <6 x float>* %n, i64 2, i1 false, i32 3, i32 3)
  call void @llvm.matrix.column.major.store.v6f32.p0v6f32(<6 x float> zeroinitializer, <6 x float>* %n, i64 %arg, i1 false, i32 3, i32 3)
  ret void
}
