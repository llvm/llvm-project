; Check that after tfr-cleanup COPY to $r0 is converted to tfrsi instruction
; The tfrst instruction must use the same slot index as the COPY instruction
; to avoid breaking live interval information.
; Check that there is no machine verifier crash

; RUN: llc -stop-after=tfr-cleanup -verify-machineinstrs %s -o - | FileCheck %s

; CHECK: $r0 = A2_tfrsi 34767

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: nounwind optsize
define dso_local i32 @foo() local_unnamed_addr #0 {
entry:
  call void @bar(i32 34767) #1
  call void @baz(i32 34767) #1
  ret i32 15
}

declare void @bar(i32) local_unnamed_addr

declare void @baz(i32) local_unnamed_addr

attributes #0 = { nounwind optsize "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv68" "target-features"="+v68,-long-calls" }
attributes #1 = { noduplicate nomerge nounwind }
