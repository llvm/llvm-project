; Test 32-bit unsigned comparison in which the second operand is a variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register comparison.
define double @f1(double %a, double %b, i32 %i1, i32 %i2) {
; CHECK-LABEL: f1:
; CHECK: clrbl %r2, %r3, 0(%r14)
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CL range.
define double @f2(double %a, double %b, i32 %i1, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: cl %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned CL range.
define double @f3(double %a, double %b, i32 %i1, ptr %base) {
; CHECK-LABEL: f3:
; CHECK: cl %r2, 4092(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1023
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which should use CLY instead of CL.
define double @f4(double %a, double %b, i32 %i1, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: cly %r2, 4096(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1024
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned CLY range.
define double @f5(double %a, double %b, i32 %i1, ptr %base) {
; CHECK-LABEL: f5:
; CHECK: cly %r2, 524284(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 131071
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f6(double %a, double %b, i32 %i1, ptr %base) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, 524288
; CHECK: cl %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 131072
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative aligned CLY range.
define double @f7(double %a, double %b, i32 %i1, ptr %base) {
; CHECK-LABEL: f7:
; CHECK: cly %r2, -4(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 -1
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CLY range.
define double @f8(double %a, double %b, i32 %i1, ptr %base) {
; CHECK-LABEL: f8:
; CHECK: cly %r2, -524288(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 -131072
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f9(double %a, double %b, i32 %i1, ptr %base) {
; CHECK-LABEL: f9:
; CHECK: agfi %r3, -524292
; CHECK: cl %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 -131073
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CL allows an index.
define double @f10(double %a, double %b, i32 %i1, i64 %base, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: cl %r2, 4092({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLY allows an index.
define double @f11(double %a, double %b, i32 %i1, i64 %base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: cly %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %i2 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the comparison can be reversed if that allows CL to be used.
define double @f12(double %a, double %b, i32 %i2, ptr %ptr) {
; CHECK-LABEL: f12:
; CHECK: cl %r2, 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i1 = load i32, ptr %ptr
  %cond = icmp ult i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
