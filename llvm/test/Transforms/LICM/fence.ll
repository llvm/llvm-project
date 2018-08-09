; RUN: opt -licm -basicaa < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s

define void @test1(i64 %n) {
; CHECK-LABEL: @test1
; CHECK-LABEL: loop:
; CHECK: fence
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
; CHECK-LABEL: loop:
; CHECK: fence
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
; CHECK-LABEL: loop:
; CHECK: fence
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
; CHECK-LABEL: loop:
; CHECK: fence
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

define void @testneg1(i64 %n, i64* %p) {
; CHECK-LABEL: @testneg1
; CHECK-LABEL: loop:
; CHECK: fence
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  store i64 %iv, i64* %p
  fence release
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

define void @testneg2(i64* %p) {
; CHECK-LABEL: @testneg2
; CHECK-LABEL: loop:
; CHECK: fence
entry:
  br label %loop
loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  fence acquire
  %n = load i64, i64* %p
  %iv.next = add i64 %iv, 1
  %test = icmp slt i64 %iv, %n
  br i1 %test, label %loop, label %exit
exit:
  ret void
}

define void @testfp1(i64 %n, i64* %p) {
; CHECK-LABEL: @testfp1
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

