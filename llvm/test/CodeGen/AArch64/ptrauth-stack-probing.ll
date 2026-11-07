; RUN: llc < %s -mtriple=arm64e-apple-darwin -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=arm64e-apple-darwin -verify-machineinstrs -O0 -global-isel | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64e-apple-darwin"

declare i32 @use_ptr(i32*)

define void @test_chkstk_ptrauth() #0 {
; CHECK-LABEL: _test_chkstk_ptrauth
; CHECK:  mov	w9, #16384
; CHECK:  adrp    x17, l___chkstk_darwin$auth_ptr$ia$0$addr@PAGE
; CHECK:  add     x17, x17, l___chkstk_darwin$auth_ptr$ia$0$addr@PAGEOFF
; CHECK:  ldr     x16, [x17]
; CHECK:  blraa   x16, x17
; CHECK-NEXT:  sub sp, sp, #4, lsl #12
  %1 = alloca [4096 x i32], align 4
  %2 = getelementptr inbounds [4096 x i32], [4096 x i32]* %1, i32 0, i32 0
  %3 = call i32 @use_ptr(i32* %2)
  ret void
}



attributes #0 = { "ptrauth-calls" "probe-stack"="__chkstk_darwin" }


; CHECK: .section  __DATA,__auth_ptr
; CHECK: .p2align  3
; CHECK: l___chkstk_darwin$auth_ptr$ia$0$addr:
; CHECK: .quad ___chkstk_darwin@AUTH(ia,0,addr)
