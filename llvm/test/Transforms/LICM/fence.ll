; RUN: opt -passes=licm < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

define void @test1(i64 %n) {
; CHECK-LABEL: @test1
; CHECK: fence
; CHECK-LABEL: loop:
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  fence release
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

define void @test2(i64 %n) {
; CHECK-LABEL: @test2
; CHECK: fence
; CHECK-LABEL: loop:
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  fence acquire
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

define void @test3(i64 %n) {
; CHECK-LABEL: @test3
; CHECK: fence
; CHECK-LABEL: loop:
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  fence acq_rel
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

define void @test4(i64 %n) {
; CHECK-LABEL: @test4
; CHECK: fence
; CHECK-LABEL: loop:
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  fence seq_cst
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

define void @testneg1(i64 %n, ptr %p) {
; CHECK-LABEL: @testneg1
; CHECK-LABEL: loop:
; CHECK: fence
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  store i64 %iv, ptr %p
  fence release
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

define void @testneg2(ptr %p) {
; CHECK-LABEL: @testneg2
; CHECK-LABEL: loop:
; CHECK: fence
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  fence acquire
  %n = load i64, ptr %p
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

; Note: While a false negative for LICM on it's own, O3 does get this
; case by combining the fences.
define void @testfn1(i64 %n, ptr %p) {
; CHECK-LABEL: @testfn1
; CHECK-LABEL: loop:
; CHECK: fence
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  fence release
  fence release
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

