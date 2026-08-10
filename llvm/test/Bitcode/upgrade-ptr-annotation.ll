; Test upgrade of ptr.annotation intrinsics.
;
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s

; Unused return values
; The arguments passed to the intrinisic wouldn't normally be arguments to
; the function, but that makes it easier to test that they are handled
; correctly.
define void @f1(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3) {
;CHECK: @f1(ptr [[ARG0:%.*]], ptr [[ARG1:%.*]], ptr [[ARG2:%.*]], i32 [[ARG3:%.*]])
  %t0 = call i8* @llvm.ptr.annotation.p0i8(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3)
;CHECK:  call ptr @llvm.ptr.annotation.p0.p0(ptr [[ARG0]], ptr [[ARG1]], ptr [[ARG2]], i32 [[ARG3]], ptr null)

  %arg0_p16 = bitcast i8* %arg0 to i16*
  %t1 = call i16* @llvm.ptr.annotation.p0i16(i16* %arg0_p16, i8* %arg1, i8* %arg2, i32 %arg3)
;CHECK:  [[ARG0_P16:%.*]] = bitcast
;CHECK:  call ptr @llvm.ptr.annotation.p0.p0(ptr [[ARG0_P16]], ptr [[ARG1]], ptr [[ARG2]], i32 [[ARG3]], ptr null)

  %arg0_p256 = bitcast i8* %arg0 to i256*
  %t2 = call i256* @llvm.ptr.annotation.p0i256(i256* %arg0_p256, i8* %arg1, i8* %arg2, i32 %arg3)
;CHECK:  [[ARG0_P256:%.*]] = bitcast
;CHECK:  call ptr @llvm.ptr.annotation.p0.p0(ptr [[ARG0_P256]], ptr [[ARG1]], ptr [[ARG2]], i32 [[ARG3]], ptr null)
  ret void
}

; Used return values
define i16* @f2(i16* %x, i16* %y) {
  %t0 = call i16* @llvm.ptr.annotation.p0i16(i16* %x, i8* undef, i8* undef, i32 undef)
  %t1 = call i16* @llvm.ptr.annotation.p0i16(i16* %y, i8* undef, i8* undef, i32 undef)
  %cmp = icmp ugt i16* %t0, %t1
  %sel = select i1 %cmp, i16* %t0, i16* %t1
  ret i16* %sel
; CHECK:  [[T0:%.*]] = call ptr @llvm.ptr.annotation.p0.p0(ptr %x, ptr undef, ptr undef, i32 undef, ptr null)
; CHECK:  [[T1:%.*]] = call ptr @llvm.ptr.annotation.p0.p0(ptr %y, ptr undef, ptr undef, i32 undef, ptr null)
; CHECK:  %cmp = icmp ugt ptr [[T0]], [[T1]]
; CHECK:  %sel = select i1 %cmp, ptr [[T0]], ptr [[T1]]
; CHECK:  ret ptr %sel
}

; CHECK: declare ptr   @llvm.ptr.annotation.p0.p0(ptr, ptr, ptr, i32, ptr)
declare i8*   @llvm.ptr.annotation.p0i8(i8*, i8*, i8*, i32)
declare i16*  @llvm.ptr.annotation.p0i16(i16*, i8*, i8*, i32)
declare i256* @llvm.ptr.annotation.p0i256(i256*, i8*, i8*, i32)
