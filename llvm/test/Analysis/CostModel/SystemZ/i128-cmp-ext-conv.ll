; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;

define i128 @fun1(i128 %val1, i128 %val2) {
; CHECK-LABEL: 'fun1'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %cmp = icmp eq i128 %val1, %val2
; CHECK: Cost Model: Found an estimated cost of 5 for instruction:   %v128 = sext i1 %cmp to i128
  %cmp = icmp eq i128 %val1, %val2
  %v128 = sext i1 %cmp to i128
  ret i128 %v128
}

define i128 @fun2(i128 %val1, i128 %val2) {
; CHECK-LABEL: 'fun2'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %cmp = icmp eq i128 %val1, %val2
; CHECK: Cost Model: Found an estimated cost of 5 for instruction:   %v128 = zext i1 %cmp to i128
  %cmp = icmp eq i128 %val1, %val2
  %v128 = zext i1 %cmp to i128
  ret i128 %v128
}

define i128 @fun3(i128 %val1, i128 %val2,
                  i128 %val3, i128 %val4) {
; CHECK-LABEL: 'fun3'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %cmp = icmp eq i128 %val1, %val2
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %add = add i128 %val3, %val4
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %sel = select i1 %cmp, i128 %val3, i128 %add
  %cmp = icmp eq i128 %val1, %val2
  %add = add i128 %val3, %val4
  %sel = select i1 %cmp, i128 %val3, i128 %add
  ret i128 %sel
}

define i128 @fun4(ptr %src) {
; CHECK-LABEL: 'fun4'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = sext i64 %v to i128
  %v = load i64, ptr %src, align 8
  %res = sext i64 %v to i128
  ret i128 %res
}

define i128 @fun5(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: 'fun5'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = sext i64 %v to i128
  %v = add i64 %lhs, %rhs
  %res = sext i64 %v to i128
  ret i128 %res
}

define i128 @fun6(ptr %src) {
; CHECK-LABEL: 'fun6'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res = zext i64 %v to i128
  %v = load i64, ptr %src, align 8
  %res = zext i64 %v to i128
  ret i128 %res
}

define i128 @fun7(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: 'fun7'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = zext i64 %v to i128
  %v = add i64 %lhs, %rhs
  %res = zext i64 %v to i128
  ret i128 %res
}

; Truncating store is free.
define void @fun8(i128 %lhs, i128 %rhs, ptr %dst) {
; CHECK-LABEL: 'fun8'
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %t = trunc i128 %v to i64
  %v = add i128 %lhs, %rhs
  %t = trunc i128 %v to i64
  store i64 %t, ptr %dst, align 8
  ret void
}

; If there is a non-store user, an extraction is needed.
define i64 @fun9(i128 %lhs, i128 %rhs, ptr %dst) {
; CHECK-LABEL: 'fun9'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %t = trunc i128 %v to i64
  %v = add i128 %lhs, %rhs
  %t = trunc i128 %v to i64
  store i64 %t, ptr %dst, align 8
  ret i64 %t
}

; Truncation of load is free.
define i64 @fun10(ptr %src) {
; CHECK-LABEL: 'fun10'
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %t = trunc i128 %v to i64
  %v = load i128, ptr %src, align 8
  %t = trunc i128 %v to i64
  ret i64 %t
}

; If the load has another user, the truncation becomes an extract.
define i64 @fun11(ptr %src, i128 %val2, ptr %dst) {
; CHECK-LABEL: 'fun11'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %t = trunc i128 %v to i64
  %v = load i128, ptr %src, align 8
  %t = trunc i128 %v to i64
  %a = add i128 %v, %val2
  store i128 %a, ptr %dst
  ret i64 %t
}

; Trunction with a GPR use typically requires an extraction.
define i64 @fun12(i128 %lhs, i128 %rhs) {
; CHECK-LABEL: 'fun12'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %t = trunc i128 %v to i64
  %v = add i128 %lhs, %rhs
  %t = trunc i128 %v to i64
  ret i64 %t
}

; Fp<->Int conversions require libcalls.
define void @fun13() {
; CHECK-LABEL: 'fun13'
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v0 = fptosi fp128 undef to i128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v1 = fptosi double undef to i128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v2 = fptosi float undef to i128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v3 = fptoui fp128 undef to i128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v4 = fptoui double undef to i128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v5 = fptoui float undef to i128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v6 = sitofp i128 undef to fp128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v7 = sitofp i128 undef to double
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v8 = sitofp i128 undef to float
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v9 = uitofp i128 undef to fp128
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v10 = uitofp i128 undef to double
; CHECK: Cost Model: Found an estimated cost of 30 for instruction:   %v11 = uitofp i128 undef to float
  %v0 = fptosi fp128 undef to i128
  %v1 = fptosi double undef to i128
  %v2 = fptosi float undef to i128
  %v3 = fptoui fp128 undef to i128
  %v4 = fptoui double undef to i128
  %v5 = fptoui float undef to i128
  %v6 = sitofp i128 undef to fp128
  %v7 = sitofp i128 undef to double
  %v8 = sitofp i128 undef to float
  %v9 = uitofp i128 undef to fp128
  %v10 = uitofp i128 undef to double
  %v11 = uitofp i128 undef to float
  ret void
}
