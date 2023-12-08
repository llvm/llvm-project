; Test byteswaps between registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.bswap.i32(i32 %a)
declare i64 @llvm.bswap.i64(i64 %a)

; Check 32-bit register-to-register byteswaps.
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: lrvr [[REGISTER:%r[0-5]]], %r2
; CHECK: br %r14
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check 64-bit register-to-register byteswaps.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: lrvgr %r2, %r2
; CHECK: br %r14
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}
