; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT:  %phi.h = phi i32 [ 0, %entry ], [ %inc, %C ], [ %inc, %D ], [ %inc, %E ]
; CHECK: DIVERGENT:       %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:       %div.cond = icmp slt i32 %tid, 0
; CHECK: DIVERGENT:  %inc = add i32 %phi.h, 1
; CHECK: DIVERGENT:       br i1 %div.cond, label %C, label %D

define void @nested_loop_extension() {
entry:
  %anchor = call token @llvm.experimental.convergence.anchor()
  br label %A

A:
  %phi.h = phi i32 [ 0, %entry ], [ %inc, %C ], [ %inc, %D ], [ %inc, %E ]
  br label %B

B:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %inc = add i32 %phi.h, 1
  br i1 %div.cond, label %C, label %D

C:
  br i1 undef, label %A, label %E

D:
  br i1 undef, label %A, label %E

E:
  %b = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %anchor) ]
  br i1 undef, label %A, label %F

F:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

declare token @llvm.experimental.convergence.anchor()
declare token @llvm.experimental.convergence.loop()

attributes #0 = { nounwind readnone }
