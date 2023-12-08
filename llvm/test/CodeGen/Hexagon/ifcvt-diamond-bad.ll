; RUN: llc -march=hexagon -minimum-jump-tables=1 < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon"

%struct.t0 = type { i8, [2 x i8] }
%struct.t1 = type { i8, i8, [1900 x i8], %struct.t0 }

@var = internal global [3 x %struct.t1] zeroinitializer, align 8
declare void @foo() #2
declare void @bar(i32, i32) #2

; Function Attrs: nounwind
define void @fred(i8 signext %a, i8 signext %b) #1 {
entry:
  %i = sext i8 %a to i32
  %t = getelementptr inbounds [3 x %struct.t1], ptr @var, i32 0, i32 %i, i32 3, i32 0
  %0 = load i8, ptr %t, align 8
  switch i8 %0, label %if.end14 [
    i8 1, label %if.then
    i8 0, label %do.body
  ]

if.then:                                          ; preds = %entry
  %j = sext i8 %b to i32
  %u = getelementptr inbounds [3 x %struct.t1], ptr @var, i32 0, i32 %i, i32 3, i32 1, i32 %j
  store i8 1, ptr %u, align 1
  tail call void @foo() #0
  br label %if.end14

do.body:                                          ; preds = %entry
  %conv11 = sext i8 %b to i32
  tail call void @bar(i32 %i, i32 %conv11) #0
  br label %if.end14

if.end14:                                         ; preds = %entry, %do.body, %if.then
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "disable-tail-calls"="false" }
attributes #2 = { "disable-tail-calls"="false" }
