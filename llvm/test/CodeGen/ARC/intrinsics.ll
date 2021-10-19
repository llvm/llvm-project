; RUN: llc -march=arc < %s | FileCheck %s

target triple = "arc"

declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.readcyclecounter()

; CHECK-LABEL: test_ctlz_i32:
; CHECK:       fls.f   %r0, %r0
; CHECK-NEXT:  mov.eq  %r0, 32
; CHECK-NEXT:  rsub.ne %r0, %r0, 31
define i32 @test_ctlz_i32(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: test_cttz_i32:
; CHECK:       ffs.f   %r0, %r0
; CHECK-NEXT:  mov.eq  %r0, 32
define i32 @test_cttz_i32(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: test_readcyclecounter:
; CHECK:       lr %r0, [33]
; CHECK-NEXT:  mov %r1, 0
define i64 @test_readcyclecounter() nounwind {
  %a = call i64 @llvm.readcyclecounter()
  ret i64 %a
}
