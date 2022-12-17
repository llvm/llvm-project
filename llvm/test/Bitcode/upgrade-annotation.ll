; Test upgrade of llvm.annotation intrinsics.
;
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-dis --opaque-pointers=0 < %s.bc | FileCheck %s --check-prefix=TYPED
; RUN: llvm-dis --opaque-pointers=1 < %s.bc | FileCheck %s


; TYPED: define i32 @f(i32 [[ARG0:%.*]], i8* [[ARG1:%.*]], i8* [[ARG2:%.*]], i32 [[ARG3:%.*]])
; CHECK: define i32 @f(i32 [[ARG0:%.*]], ptr [[ARG1:%.*]], ptr [[ARG2:%.*]], i32 [[ARG3:%.*]])
define i32 @f(i32 %arg0, ptr %arg1, ptr %arg2, i32 %arg3) {
  %result = call i32 @llvm.annotation.i32(i32 %arg0, ptr %arg1, ptr %arg2, i32 %arg3)
  ; TYPED: [[RESULT:%.*]] = call i32 @llvm.annotation.i32.p0i8(i32 [[ARG0]], i8* [[ARG1]], i8* [[ARG2]], i32 [[ARG3]])
  ; CHECK: [[RESULT:%.*]] = call i32 @llvm.annotation.i32.p0(i32 [[ARG0]], ptr [[ARG1]], ptr [[ARG2]], i32 [[ARG3]])
  ret i32 %result
}

declare i32 @llvm.annotation.i32(i32, i8*, ptr, i32)
; TYPED: declare i32 @llvm.annotation.i32.p0i8(i32, i8*, i8*, i32)
; CHECK: declare i32 @llvm.annotation.i32.p0(i32, ptr, ptr, i32)
