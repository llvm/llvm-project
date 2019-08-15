; RUN: llc -mtriple=x86_64-apple-macosx -stop-after=finalize-isel %s -o - | FileCheck %s

declare i8* @llvm.ptrmask.p0i8.i64(i8* , i64)

; CHECK-LABEL: name: test1
; CHECK:         %0:gr64 = COPY $rdi
; CHECK-NEXT:    %1:gr64 = MOV64ri 72057594037927928
; CHECK-NEXT:    %2:gr64 = AND64rr %0, killed %1, implicit-def dead $eflags
; CHECK-NEXT:    $rax = COPY %2
; CHECK-NEXT:    RET 0, $rax

define i8* @test1(i8* %src) {
  %ptr = call i8* @llvm.ptrmask.p0i8.i64(i8* %src, i64 72057594037927928)
  ret i8* %ptr
}

declare i8* @llvm.ptrmask.p0i8.i32(i8*, i32)

; CHECK-LABEL: name: test2
; CHECK:         %0:gr64 = COPY $rdi
; CHECK-NEXT:    %1:gr32 = COPY %0.sub_32bit
; CHECK-NEXT:    %2:gr32 = AND32ri %1, 10000, implicit-def dead $eflags
; CHECK-NEXT:    %3:gr64 = SUBREG_TO_REG 0, killed %2, %subreg.sub_32bit
; CHECK-NEXT:    $rax = COPY %3
; CHECK-NEXT:    RET 0, $rax


define i8* @test2(i8* %src) {
  %ptr = call i8* @llvm.ptrmask.p0i8.i32(i8* %src, i32 10000)
  ret i8* %ptr
}
