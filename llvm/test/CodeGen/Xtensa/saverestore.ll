; RUN: llc --mtriple=xtensa < %s | FileCheck %s

declare ptr @llvm.stacksave()

declare void @llvm.stackrestore(ptr)

declare void @use_addr(ptr)

define void @test_saverestore(i64 %n) {
; CHECK:       # %bb.0:
; CHECK-NEXT:  addi a8, a1, -16
; CHECK-NEXT:  or a1, a8, a8
; CHECK:       s32i a0, a1, 8
; CHECK-NEXT:  s32i a12, a1, 4
; CHECK-NEXT:  s32i a15, a1, 0
; CHECK:       or a15, a1, a1
; CHECK:       addi a8, a2, 3
; CHECK-NEXT:  movi a9, -4
; CHECK-NEXT:  and a8, a8, a9
; CHECK-NEXT:  addi a8, a8, 31
; CHECK-NEXT:  movi a9, -32
; CHECK-NEXT:  and a8, a8, a9
; CHECK-NEXT:  or a12, a1, a1
; CHECK-NEXT:  sub a1, a1, a8
; CHECK-NEXT:  or a2, a1, a1
; CHECK-NEXT:  l32r a8, .LCPI0_0
; CHECK-NEXT:  callx0 a8
; CHECK-NEXT:  or a1, a12, a12
; CHECK-NEXT:  or a1, a15, a15
; CHECK-NEXT:  l32i a15, a1, 0
; CHECK-NEXT:  l32i a12, a1, 4
; CHECK-NEXT:  l32i a0, a1, 8
; CHECK-NEXT:  addi a8, a1, 16
; CHECK-NEXT:  or a1, a8, a8
; CHECK-NEXT:  ret

  %sp = call ptr @llvm.stacksave.p0()
  %addr = alloca i8, i64 %n
  call void @use_addr(ptr %addr)
  call void @llvm.stackrestore.p0(ptr %sp)
  ret void
}
