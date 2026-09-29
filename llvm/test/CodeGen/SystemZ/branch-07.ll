; Test all condition-code masks that are relevant for CGRJ.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

; Test EQ.
define void @f1(i64 %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrje %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp eq i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Test NE.
define void @f2(i64 %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjlh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp ne i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Test SLE.
define void @f3(i64 %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjle %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp sle i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Test SLT.
define void @f4(i64 %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjl %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp slt i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Test SGT.
define void @f5(i64 %target) {
; CHECK-LABEL: f5:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp sgt i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Test SGE.
define void @f6(i64 %target) {
; CHECK-LABEL: f6:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjhe %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp sge i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Test a vector of 0/-1 results for i32 EQ.
define i64 @f7(i64 %a, i64 %b) {
; CHECK-LABEL: f7:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], -268435456
; CHECK: sra [[REG]], 31
; CHECK: br %r14
  %avec = bitcast i64 %a to <2 x i32>
  %bvec = bitcast i64 %b to <2 x i32>
  %cmp = icmp eq <2 x i32> %avec, %bvec
  %ext = sext <2 x i1> %cmp to <2 x i32>
  %ret = bitcast <2 x i32> %ext to i64
  ret i64 %ret
}

; Test a vector of 0/-1 results for i32 NE.
define i64 @f8(i64 %a, i64 %b) {
; CHECK-LABEL: f8:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1879048192
; CHECK: sra [[REG]], 31
; CHECK: br %r14
  %avec = bitcast i64 %a to <2 x i32>
  %bvec = bitcast i64 %b to <2 x i32>
  %cmp = icmp ne <2 x i32> %avec, %bvec
  %ext = sext <2 x i1> %cmp to <2 x i32>
  %ret = bitcast <2 x i32> %ext to i64
  ret i64 %ret
}

; Test a vector of 0/-1 results for i64 EQ.
define void @f9(i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f9:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], -268435456
; CHECK: sllg [[REG2:%r[0-5]]], [[REG]], 32
; CHECK: srag {{%r[0-5]}}, [[REG2]], 63
; CHECK: br %r14
  %avec = bitcast i64 %a to <2 x i32>
  %bvec = bitcast i64 %b to <2 x i32>
  %cmp = icmp eq <2 x i32> %avec, %bvec
  %ext = sext <2 x i1> %cmp to <2 x i64>
  store <2 x i64> %ext, ptr %dest
  ret void
}

; Test a vector of 0/-1 results for i64 NE.
define void @f10(i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f10:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1879048192
; CHECK: sllg [[REG2:%r[0-5]]], [[REG]], 32
; CHECK: srag {{%r[0-5]}}, [[REG2]], 63
; CHECK: br %r14
  %avec = bitcast i64 %a to <2 x i32>
  %bvec = bitcast i64 %b to <2 x i32>
  %cmp = icmp ne <2 x i32> %avec, %bvec
  %ext = sext <2 x i1> %cmp to <2 x i64>
  store <2 x i64> %ext, ptr %dest
  ret void
}
