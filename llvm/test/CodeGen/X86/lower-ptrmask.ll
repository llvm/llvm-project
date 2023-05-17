; RUN: llc -mtriple=x86_64-apple-macosx -stop-after=finalize-isel %s -o - | FileCheck %s

declare ptr @llvm.ptrmask.p0.i64(ptr , i64)

; CHECK-LABEL: name: test1
; CHECK:         %0:gr64 = COPY $rdi
; CHECK-NEXT:    %1:gr64 = MOV64ri 72057594037927928
; CHECK-NEXT:    %2:gr64 = AND64rr %0, killed %1, implicit-def dead $eflags
; CHECK-NEXT:    $rax = COPY %2
; CHECK-NEXT:    RET 0, $rax

define ptr @test1(ptr %src) {
  %ptr = call ptr @llvm.ptrmask.p0.i64(ptr %src, i64 72057594037927928)
  ret ptr %ptr
}

declare ptr @llvm.ptrmask.p0.i32(ptr, i32)

; CHECK-LABEL: name: test2
; CHECK:         %0:gr64 = COPY $rdi
; CHECK-NEXT:    %1:gr32 = COPY %0.sub_32bit
; CHECK-NEXT:    %2:gr32 = AND32ri %1, 10000, implicit-def dead $eflags
; CHECK-NEXT:    %3:gr64 = SUBREG_TO_REG 0, killed %2, %subreg.sub_32bit
; CHECK-NEXT:    $rax = COPY %3
; CHECK-NEXT:    RET 0, $rax


define ptr @test2(ptr %src) {
  %ptr = call ptr @llvm.ptrmask.p0.i32(ptr %src, i32 10000)
  ret ptr %ptr
}
