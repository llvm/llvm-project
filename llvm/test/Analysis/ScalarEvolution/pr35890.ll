; RUN: opt < %s -scalar-evolution-max-arith-depth=0  -passes=indvars  -S | FileCheck %s

target datalayout = "e-m:e-i32:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; Check that it does not crash because SCEVAddRec's step is not an AddRec.

define void @pr35890(i32* %inc_ptr, i32 %a) {

; CHECK-LABEL: @pr35890(

entry:
  %inc = load i32, i32* %inc_ptr, !range !0
  %ne.cond = icmp ne i32 %inc, 0
  br i1 %ne.cond, label %loop, label %bail

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %a.1 = add i32 %a, 1
  %iv.next = add i32 %iv, %a.1
  %iv.wide = zext i32 %iv to i64
  %iv.square = mul i64 %iv.wide, %iv.wide
  %iv.cube = mul i64 %iv.square, %iv.wide
  %brcond = icmp slt i64 %iv.wide, %iv.cube
  br i1 %brcond, label %if.true, label %if.false

if.true:
  br label %backedge

if.false:
  br label %backedge

backedge:
  %loopcond = icmp slt i32 %iv, 200
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void

bail:
  ret void
}

!0 = !{i32 0, i32 100}
