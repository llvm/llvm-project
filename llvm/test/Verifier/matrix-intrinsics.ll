; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare <4 x float> @llvm.matrix.transpose.v4f32(<4 x float>, i32, i32)
define <4 x float> @transpose(<4 x float> %m) {
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
  %result.1 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %m, i32 3, i32 2)
  %result.2 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.1, i32 2, i32 1)
  ret <4 x float> %result.2
}

declare <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float>, <4 x float>, i32, i32, i32)
define <4 x float> @multiply(<4 x float> %m) {
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
  %result.1 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %m, <4 x float> %m, i32 3, i32 2, i32 2)
  %result.2 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.1, <4 x float> %m, i32 2, i32 2, i32 1)
  ret <4 x float> %result.2
}

declare <4 x float> @llvm.matrix.columnwise.load.v4f32.p0v4f32(<4 x float>*, i32, i32, i32)
declare <6 x float> @llvm.matrix.columnwise.load.v6f32.p0v6f32(<6 x float>*, i32, i32, i32)
define <4 x float> @columnwise_load(<4 x float>* %m, <6 x float>* %n) {
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
  %result.1 = call <4 x float> @llvm.matrix.columnwise.load.v4f32.p0v4f32(<4 x float>* %m, i32 2, i32 1, i32 2)
  %result.2 = call <6 x float> @llvm.matrix.columnwise.load.v6f32.p0v6f32(<6 x float>* %n, i32 2, i32 3, i32 3)
  ret <4 x float> %result.1
}

declare void @llvm.matrix.columnwise.store.v4f32.p0v4f32(<4 x float>, <4 x float>*, i32, i32, i32)
declare void @llvm.matrix.columnwise.store.v6f32.p0v6f32(<6 x float>, <6 x float>*, i32, i32, i32)
define void @columnwise_store(<4 x float>* %m, <6 x float>* %n) {
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
; CHECK-NEXT: result of a matrix operation does not fit in the returned vector
  call void @llvm.matrix.columnwise.store.v4f32.p0v4f32(<4 x float> zeroinitializer, <4 x float>* %m, i32 2, i32 1, i32 2)
  call void @llvm.matrix.columnwise.store.v6f32.p0v6f32(<6 x float> zeroinitializer, <6 x float>* %n, i32 2, i32 3, i32 3)
  ret void
}
