; RUN: opt -S -passes="loop-mssa(simple-loop-unswitch)" -disable-basic-aa -verify-memoryssa < %s | FileCheck %s
; REQUIRES: asserts

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL:  @foo()
; Function Attrs: readnone speculatable
declare i32 @foo() #0

define void @main() {
entry:
  br label %for.cond2682

for.cond2682:                                     ; preds = %if.then2712, %entry
  %mul2708 = call i32 @foo()
  %tobool2709 = icmp ne i32 %mul2708, 0
  br i1 %tobool2709, label %if.then2712, label %lor.lhs.false2710

lor.lhs.false2710:                                ; preds = %for.cond2682
  unreachable

if.then2712:                                      ; preds = %for.cond2682
  br label %for.cond2682
}

attributes #0 = { readnone speculatable }
