; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

; Make sure that the subfic is generated iff possible

define i64 @subtract_from_imm1(i64 %v) nounwind readnone {
entry:
; CHECK-LABEL: subtract_from_imm1
; CHECK: subfic 3, 3, 32767
; CHECK: blr
  %sub = sub i64 32767, %v
  ret i64 %sub
}

define i64 @subtract_from_imm2(i64 %v) nounwind readnone {
entry:
; CHECK-LABEL: subtract_from_imm2
; CHECK-NOT: subfic
; CHECK: blr
  %sub = sub i64 32768, %v
  ret i64 %sub
}

define i64 @subtract_from_imm3(i64 %v) nounwind readnone {
entry:
; CHECK-LABEL: subtract_from_imm3
; CHECK: subfic 3, 3, -32768
; CHECK: blr
  %sub = sub i64 -32768, %v
  ret i64 %sub
}

define i64 @subtract_from_imm4(i64 %v) nounwind readnone {
entry:
; CHECK-LABEL: subtract_from_imm4
; CHECK-NOT: subfic
; CHECK: blr
  %sub = sub i64 -32769, %v
  ret i64 %sub
}

