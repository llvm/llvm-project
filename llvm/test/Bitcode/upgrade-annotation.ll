; Test upgrade of llvm.annotation intrinsics.
;
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s


; CHECK: define i32 @f(i32 [[ARG0:%.*]], i8* [[ARG1:%.*]], i8* [[ARG2:%.*]], i32 [[ARG3:%.*]])
define i32 @f(i32 %arg0, i8* %arg1, i8* %arg2, i32 %arg3) {
  %result = call i32 @llvm.annotation.i32(i32 %arg0, i8* %arg1, i8* %arg2, i32 %arg3)
  ; CHECK: [[RESULT:%.*]] = call i32 @llvm.annotation.i32(i32 [[ARG0]], i8* [[ARG1]], i8* [[ARG2]], i32 [[ARG3]])
  ret i32 %result
}

declare i32 @llvm.annotation.i32(i32, i8*, i8*, i32)
; CHECK: declare i32 @llvm.annotation.i32(i32, i8*, i8*, i32)
