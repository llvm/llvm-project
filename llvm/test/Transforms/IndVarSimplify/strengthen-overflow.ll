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

define hidden void @test.shl.exact.equal() {
; CHECK-LABEL: @test.shl.exact.equal
entry:
  br label %for.body

for.body:
; CHECK-LABEL: for.body
  %k.021 = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  %shl = shl i32 1, %k.021
  %shr1 = ashr i32 %shl, 1
; CHECK: %shr1 = ashr exact i32 %shl, 1
  %shr2 = lshr i32 %shl, 1
; CHECK: %shr2 = lshr exact i32 %shl, 1
  %inc = add nuw nsw i32 %k.021, 1
  %exitcond = icmp eq i32 %inc, 9
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define hidden void @test.shl.exact.greater() {
; CHECK-LABEL: @test.shl.exact.greater
entry:
  br label %for.body

for.body:
; CHECK-LABEL: for.body
  %k.021 = phi i32 [ 3, %entry ], [ %inc, %for.body ]
  %shl = shl i32 1, %k.021
  %shr1 = ashr i32 %shl, 2
; CHECK: %shr1 = ashr exact i32 %shl, 2
  %shr2 = lshr i32 %shl, 2
; CHECK: %shr2 = lshr exact i32 %shl, 2
  %inc = add nuw nsw i32 %k.021, 1
  %exitcond = icmp eq i32 %inc, 9
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define hidden void @test.shl.exact.unbound(i32 %arg) {
; CHECK-LABEL: @test.shl.exact.unbound
entry:
  br label %for.body

for.body:
; CHECK-LABEL: for.body
  %k.021 = phi i32 [ 2, %entry ], [ %inc, %for.body ]
  %shl = shl i32 1, %k.021
  %shr1 = ashr i32 %shl, 2
; CHECK: %shr1 = ashr exact i32 %shl, 2
  %shr2 = lshr i32 %shl, 2
; CHECK: %shr2 = lshr exact i32 %shl, 2
  %inc = add nuw nsw i32 %k.021, 1
  %exitcond = icmp eq i32 %inc, %arg
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define hidden void @test.shl.nonexact() {
; CHECK-LABEL: @test.shl.nonexact
entry:
  br label %for.body

for.body:
; CHECK-LABEL: for.body
  %k.021 = phi i32 [ 2, %entry ], [ %inc, %for.body ]
  %shl = shl i32 1, %k.021
  %shr1 = ashr i32 %shl, 3
; CHECK: %shr1 = ashr i32 %shl, 3
  %shr2 = lshr i32 %shl, 3
; CHECK: %shr2 = lshr i32 %shl, 3
  %inc = add nuw nsw i32 %k.021, 1
  %exitcond = icmp eq i32 %inc, 9
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

!0 = !{i32 0, i32 2}
!1 = !{i32 0, i32 42}
