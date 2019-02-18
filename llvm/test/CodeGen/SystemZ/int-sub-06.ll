; Test 128-bit addition in which the second operand is a zero-extended i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register additions.  The XOR ensures that we don't instead zero-extend
; %b into a register and use memory addition.
define void @f1(i128 *%aptr, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: slgfr {{%r[0-5]}}, %r3
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Like f1, but using an "in-register" extension.
define void @f2(i128 *%aptr, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: slgfr {{%r[0-5]}}, %r3
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %trunc = trunc i64 %b to i32
  %bext = zext i32 %trunc to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Test register addition in cases where the second operand is zero extended
; from i64 rather than i32, but is later masked to i32 range.
define void @f3(i128 *%aptr, i64 %b) {
; CHECK-LABEL: f3:
; CHECK: slgfr {{%r[0-5]}}, %r3
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %bext = zext i64 %b to i128
  %and = and i128 %bext, 4294967295
  %sub = sub i128 %xor, %and
  store i128 %sub, i128 *%aptr
  ret void
}

; Test SLGF with no offset.
define void @f4(i128 *%aptr, i32 *%bsrc) {
; CHECK-LABEL: f4:
; CHECK: slgf {{%r[0-5]}}, 0(%r3)
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %b = load i32, i32 *%bsrc
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Check the high end of the SLGF range.
define void @f5(i128 *%aptr, i32 *%bsrc) {
; CHECK-LABEL: f5:
; CHECK: slgf {{%r[0-5]}}, 524284(%r3)
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %ptr = getelementptr i32, i32 *%bsrc, i64 131071
  %b = load i32, i32 *%ptr
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Check the next word up, which must use separate address logic.
; Other sequences besides this one would be OK.
define void @f6(i128 *%aptr, i32 *%bsrc) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, 524288
; CHECK: slgf {{%r[0-5]}}, 0(%r3)
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %ptr = getelementptr i32, i32 *%bsrc, i64 131072
  %b = load i32, i32 *%ptr
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Check the high end of the negative aligned SLGF range.
define void @f7(i128 *%aptr, i32 *%bsrc) {
; CHECK-LABEL: f7:
; CHECK: slgf {{%r[0-5]}}, -4(%r3)
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %ptr = getelementptr i32, i32 *%bsrc, i128 -1
  %b = load i32, i32 *%ptr
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Check the low end of the SLGF range.
define void @f8(i128 *%aptr, i32 *%bsrc) {
; CHECK-LABEL: f8:
; CHECK: slgf {{%r[0-5]}}, -524288(%r3)
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %ptr = getelementptr i32, i32 *%bsrc, i128 -131072
  %b = load i32, i32 *%ptr
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f9(i128 *%aptr, i32 *%bsrc) {
; CHECK-LABEL: f9:
; CHECK: agfi %r3, -524292
; CHECK: slgf {{%r[0-5]}}, 0(%r3)
; CHECK: slbgr
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %ptr = getelementptr i32, i32 *%bsrc, i128 -131073
  %b = load i32, i32 *%ptr
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}

; Check that SLGF allows an index.
define void @f10(i128 *%aptr, i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: slgf {{%r[0-5]}}, 524284({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %a = load i128, i128 *%aptr
  %xor = xor i128 %a, 127
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32, i32 *%ptr
  %bext = zext i32 %b to i128
  %sub = sub i128 %xor, %bext
  store i128 %sub, i128 *%aptr
  ret void
}
