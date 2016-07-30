; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner -verify-machineinstrs < %s | FileCheck %s

; If the trip count is a compile-time constant, then decrement it instead
; of computing a new LC0 value.

; CHECK-LABEL: @test
; CHECK: loop0(.LBB0_1, #998)

define i32 @test(i32* %A, i32* %B, i32 %count) {
entry:
  br label %for.body

for.body:
  %sum.02 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx.phi = phi i32* [ %A, %entry ], [ %arrayidx.inc, %for.body ]
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %add = add nsw i32 %0, %sum.02
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 1000
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %add
}

; The constant trip count is small enough that the kernel is not executed.

; CHECK-LABEL: @test1
; CHECK-NOT: loop0(

define i32 @test1(i32* %A, i32* %B, i32 %count) {
entry:
  br label %for.body

for.body:
  %sum.02 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx.phi = phi i32* [ %A, %entry ], [ %arrayidx.inc, %for.body ]
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %add = add nsw i32 %0, %sum.02
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 1
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %add
}

