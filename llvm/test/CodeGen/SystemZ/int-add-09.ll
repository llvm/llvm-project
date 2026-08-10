; Test 128-bit addition in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check additions of 1.  The XOR ensures that we don't instead load the
; constant into a register and use memory addition.
define void @f1(ptr %aptr) {
; CHECK-LABEL: f1:
; CHECK: algfi {{%r[0-5]}}, 1
; CHECK: alcg
; CHECK: br %r14
  %a = load i128, ptr %aptr
  %xor = xor i128 %a, 128
  %add = add i128 %xor, 1
  store i128 %add, ptr %aptr
  ret void
}

; Check the high end of the ALGFI range.
define void @f2(ptr %aptr) {
; CHECK-LABEL: f2:
; CHECK: algfi {{%r[0-5]}}, 4294967295
; CHECK: alcg
; CHECK: br %r14
  %a = load i128, ptr %aptr
  %xor = xor i128 %a, 128
  %add = add i128 %xor, 4294967295
  store i128 %add, ptr %aptr
  ret void
}

; Check the next value up, which must use register addition.
define void @f3(ptr %aptr) {
; CHECK-LABEL: f3:
; CHECK: algr
; CHECK: alcg
; CHECK: br %r14
  %a = load i128, ptr %aptr
  %xor = xor i128 %a, 128
  %add = add i128 %xor, 4294967296
  store i128 %add, ptr %aptr
  ret void
}

; Check addition of -1, which must also use register addition.
define void @f4(ptr %aptr) {
; CHECK-LABEL: f4:
; CHECK: algr
; CHECK: alcg
; CHECK: br %r14
  %a = load i128, ptr %aptr
  %xor = xor i128 %a, 128
  %add = add i128 %xor, -1
  store i128 %add, ptr %aptr
  ret void
}
