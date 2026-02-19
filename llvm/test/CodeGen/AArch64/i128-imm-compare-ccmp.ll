; RUN: llc -mtriple=aarch64-linux-gnu -O3 -o - < %s | FileCheck %s

define i1 @eq_imm(i128 %x) {
; CHECK-LABEL: eq_imm:
; CHECK:       cmp x0, #5
; CHECK-NEXT:  ccmp x1, #0, #0, eq
; CHECK-NEXT:  cset w0, eq
; CHECK-NEXT:  ret
  %cmp = icmp eq i128 %x, 5
  ret i1 %cmp
}

define i1 @ne_imm(i128 %x) {
; CHECK-LABEL: ne_imm:
; CHECK:       cmp x0, #5
; CHECK-NEXT:  ccmp x1, #0, #0, eq
; CHECK-NEXT:  cset w0, ne
; CHECK-NEXT:  ret
  %cmp = icmp ne i128 %x, 5
  ret i1 %cmp
}

define i1 @ult_imm(i128 %x) {
; CHECK-LABEL: ult_imm:
; CHECK:       cmp x1, #0
; CHECK-NEXT:  ccmp x0, #5, #2, eq
; CHECK-NEXT:  cset w0, {{cc|lo}}
; CHECK-NEXT:  ret
  %cmp = icmp ult i128 %x, 5
  ret i1 %cmp
}

define i1 @ule_imm(i128 %x) {
; CHECK-LABEL: ule_imm:
; CHECK:       cmp x1, #0
; CHECK-NEXT:  ccmp x0, #6, #2, eq
; CHECK-NEXT:  cset w0, {{cc|lo}}
; CHECK-NEXT:  ret
  %cmp = icmp ule i128 %x, 5
  ret i1 %cmp
}

define i1 @ugt_imm(i128 %x) {
; CHECK-LABEL: ugt_imm:
; CHECK:       cmp x1, #0
; CHECK-NEXT:  ccmp x0, #5, #2, eq
; CHECK-NEXT:  cset	w0, hi
; CHECK-NEXT:  ret
  %cmp = icmp ugt i128 %x, 5
  ret i1 %cmp
}

define i1 @uge_imm(i128 %x) {
; CHECK-LABEL: uge_imm:
; CHECK:       cmp	x1, #0
; CHECK-NEXT:  ccmp	x0, #4, #2, eq
; CHECK-NEXT:  cset	w0, hi
; CHECK-NEXT:  ret
  %cmp = icmp uge i128 %x, 5
  ret i1 %cmp
}