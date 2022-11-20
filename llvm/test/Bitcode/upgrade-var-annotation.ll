; Test upgrade of var.annotation intrinsics.
;
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s


define void @f(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3) {
;CHECK: @f(i8* [[ARG0:%.*]], i8* [[ARG1:%.*]], i8* [[ARG2:%.*]], i32 [[ARG3:%.*]])
  call void @llvm.var.annotation(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3)
;CHECK:  call void @llvm.var.annotation.p0i8.p0i8(i8* [[ARG0]], i8* [[ARG1]], i8* [[ARG2]], i32 [[ARG3]], i8* null)
  ret void
}

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.var.annotation(i8*, i8*, i8*, i32)
; CHECK: declare void @llvm.var.annotation.p0i8.p0i8(i8*, i8*, i8*, i32, i8*)
