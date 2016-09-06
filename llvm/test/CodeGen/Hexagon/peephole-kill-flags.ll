; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
; CHECK: memw

; Check that the testcase compiles without errors.

target triple = "hexagon"

; Function Attrs: nounwind
define void @fred() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %entry
  %0 = load i32, i32* undef, align 4
  %mul = mul nsw i32 2, %0
  %cmp = icmp slt i32 undef, %mul
  br i1 %cmp, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  unreachable

for.end13:                                        ; preds = %for.cond
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,-hvx-double" }

