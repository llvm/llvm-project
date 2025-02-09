; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check that we do not crash.
; CHECK: call foo

target triple = "hexagon"

%struct.0 = type { ptr, ptr, [2 x ptr], i32, i32, ptr, i32, i32, i32, i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

define i32 @fred(ptr %p0) local_unnamed_addr #0 {
entry:
  br i1 undef, label %if.then21, label %for.body.i

if.then21:                                        ; preds = %entry
  %.pr = load i32, ptr undef, align 4
  switch i32 %.pr, label %cleanup [
    i32 1, label %for.body.i
    i32 3, label %if.then60
  ]

for.body.i:                                       ; preds = %for.body.i, %if.then21, %entry
  %0 = load i8, ptr undef, align 1
  %cmp7.i = icmp ugt i8 %0, -17
  br i1 %cmp7.i, label %cleanup, label %for.body.i

if.then60:                                        ; preds = %if.then21
  %call61 = call i32 @foo(ptr nonnull %p0) #0
  br label %cleanup

cleanup:                                          ; preds = %if.then60, %for.body.i, %if.then21
  ret i32 undef
}

declare i32 @foo(ptr) local_unnamed_addr #0


attributes #0 = { nounwind }

