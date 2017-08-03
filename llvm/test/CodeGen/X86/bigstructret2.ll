; RUN: llc < %s -march=x86 -mtriple=i686-pc-linux-gnu | FileCheck %s

; CHECK: .cfi_startproc
; CHECK: .cfi_def_cfa_offset 8
; CHECK: .cfi_def_cfa_offset 12
; CHECK: .cfi_def_cfa_offset 32
; CHECK: .cfi_offset %esi, -12
; CHECK: .cfi_offset %edi, -8
; CHECK: .cfi_endproc

%0 = type { i64, i64 }

declare fastcc %0 @ReturnBigStruct() nounwind readnone

define void @test(%0* %p) {
  %1 = call fastcc %0 @ReturnBigStruct()
  store %0 %1, %0* %p
  ret void
}

