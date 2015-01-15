; RUN: opt < %s -indvars -S | FileCheck %s

define i32 @test.signed.add.0(i32* %array, i32 %length, i32 %init) {
; CHECK-LABEL: @test.signed.add.0
 entry:
  %upper = icmp slt i32 %init, %length
  br i1 %upper, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %civ = phi i32 [ %init, %entry ], [ %civ.inc, %latch ]
  %civ.inc = add i32 %civ, 1
; CHECK: %civ.inc = add nsw i32 %civ, 1
  %cmp = icmp slt i32 %civ.inc, %length
  br i1 %cmp, label %latch, label %break

 latch:
  store i32 0, i32* %array
  %check = icmp slt i32 %civ.inc, %length
  br i1 %check, label %loop, label %break

 break:
  ret i32 %civ.inc

 exit:
  ret i32 42
}

define i32 @test.signed.add.1(i32* %array, i32 %length, i32 %init) {
; CHECK-LABEL: @test.signed.add.1
 entry:
  %upper = icmp sle i32 %init, %length
  br i1 %upper, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %civ = phi i32 [ %init, %entry ], [ %civ.inc, %latch ]
  %civ.inc = add i32 %civ, 1
; CHECK: %civ.inc = add i32 %civ, 1
  %cmp = icmp slt i32 %civ.inc, %length
  br i1 %cmp, label %latch, label %break

 latch:
  store i32 0, i32* %array
  %check = icmp slt i32 %civ.inc, %length
  br i1 %check, label %loop, label %break

 break:
  ret i32 %civ.inc

 exit:
  ret i32 42
}

define i32 @test.unsigned.add.0(i32* %array, i32 %length, i32 %init) {
; CHECK-LABEL: @test.unsigned.add.0
 entry:
  %upper = icmp ult i32 %init, %length
  br i1 %upper, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %civ = phi i32 [ %init, %entry ], [ %civ.inc, %latch ]
  %civ.inc = add i32 %civ, 1
; CHECK: %civ.inc = add nuw i32 %civ, 1
  %cmp = icmp slt i32 %civ.inc, %length
  br i1 %cmp, label %latch, label %break

 latch:
  store i32 0, i32* %array
  %check = icmp ult i32 %civ.inc, %length
  br i1 %check, label %loop, label %break

 break:
  ret i32 %civ.inc

 exit:
  ret i32 42
}

define i32 @test.unsigned.add.1(i32* %array, i32 %length, i32 %init) {
; CHECK-LABEL: @test.unsigned.add.1
 entry:
  %upper = icmp ule i32 %init, %length
  br i1 %upper, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %civ = phi i32 [ %init, %entry ], [ %civ.inc, %latch ]
  %civ.inc = add i32 %civ, 1
; CHECK: %civ.inc = add i32 %civ, 1
  %cmp = icmp slt i32 %civ.inc, %length
  br i1 %cmp, label %latch, label %break

 latch:
  store i32 0, i32* %array
  %check = icmp ult i32 %civ.inc, %length
  br i1 %check, label %loop, label %break

 break:
  ret i32 %civ.inc

 exit:
  ret i32 42
}

!0 = !{i32 0, i32 2}
!1 = !{i32 0, i32 42}
