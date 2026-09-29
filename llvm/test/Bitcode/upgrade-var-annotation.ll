; Test upgrade of var.annotation intrinsics.
;
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s


define void @f(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3) {
;CHECK: @f(ptr [[ARG0:%.*]], ptr [[ARG1:%.*]], ptr [[ARG2:%.*]], i32 [[ARG3:%.*]])
  call void @llvm.var.annotation(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3)
;CHECK:  call void @llvm.var.annotation.p0.p0(ptr [[ARG0]], ptr [[ARG1]], ptr [[ARG2]], i32 [[ARG3]], ptr null)
  ret void
}

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.var.annotation(i8*, i8*, i8*, i32)
; CHECK: declare void @llvm.var.annotation.p0.p0(ptr, ptr, ptr, i32, ptr)
