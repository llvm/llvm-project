; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check that we do not crash.
; CHECK: call foo

target triple = "hexagon"

%struct.0 = type { i8*, i8*, [2 x i8*], i32, i32, i8*, i32, i32, i32, i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

define i32 @fred(i8* %p0) local_unnamed_addr #0 {
entry:
  %0 = bitcast i8* %p0 to %struct.0*
  br i1 undef, label %if.then21, label %for.body.i

if.then21:                                        ; preds = %entry
  %.pr = load i32, i32* undef, align 4
  switch i32 %.pr, label %cleanup [
    i32 1, label %for.body.i
    i32 3, label %if.then60
  ]

for.body.i:                                       ; preds = %for.body.i, %if.then21, %entry
  %1 = load i8, i8* undef, align 1
  %cmp7.i = icmp ugt i8 %1, -17
  br i1 %cmp7.i, label %cleanup, label %for.body.i

if.then60:                                        ; preds = %if.then21
  %call61 = call i32 @foo(%struct.0* nonnull %0) #0
  br label %cleanup

cleanup:                                          ; preds = %if.then60, %for.body.i, %if.then21
  ret i32 undef
}

declare i32 @foo(%struct.0*) local_unnamed_addr #0


attributes #0 = { nounwind }

