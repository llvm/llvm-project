; RUN: llc -O3 -o - -verify-machineinstrs %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct.s1 = type { %struct.s3*, %struct.s1* }
%struct.s2 = type opaque
%struct.s3 = type { i32 }

; Function Attrs: nounwind
define internal fastcc i32 @repeated_dup_worklist(%struct.s1** %pp1, %struct.s2* %p2, i32 %state, i1 %i1_1, i32 %i32_1) unnamed_addr #0 {
entry:
  br label %while.cond.outer

; The loop gets laid out:
; %while.cond.outer
; %(null)
; %(null)
; %dup2
; and then %dup1 gets chosen as the next block.
; when dup2 is duplicated into dup1, %worklist could erroneously be placed on
; the worklist, because all of its current predecessors are now scheduled.
; However, after dup2 is tail-duplicated, %worklist can't be on the worklist
; because it now has unscheduled predecessors.q
; CHECK-LABEL: repeated_dup_worklist
; CHECK: // %entry
; CHECK: // %while.cond.outer
; first %(null) block
; CHECK: // in Loop:
; CHECK: ldr
; CHECK-NEXT: tbnz
; second %(null) block
; CHECK: // in Loop:
; CHECK: // %dup2
; CHECK: // %worklist
; CHECK: // %if.then96.i
while.cond.outer:                                 ; preds = %dup1, %entry
  %progress.0.ph = phi i32 [ 0, %entry ], [ %progress.1, %dup1 ]
  %inc77 = add nsw i32 %progress.0.ph, 1
  %cmp = icmp slt i32 %progress.0.ph, %i32_1
  br i1 %cmp, label %dup2, label %dup1

dup2:                       ; preds = %if.then96.i, %worklist, %while.cond.outer
  %progress.1.ph = phi i32 [ 0, %while.cond.outer ], [ %progress.1, %if.then96.i ], [ %progress.1, %worklist ]
  %.pr = load %struct.s1*, %struct.s1** %pp1, align 8
  br label %dup1

dup1:                                       ; preds = %dup2, %while.cond.outer
  %0 = phi %struct.s1* [ %.pr, %dup2 ], [ undef, %while.cond.outer ]
  %progress.1 = phi i32 [ %progress.1.ph, %dup2 ], [ %inc77, %while.cond.outer ]
  br i1 %i1_1, label %while.cond.outer, label %worklist

worklist:                                       ; preds = %dup1
  %snode94 = getelementptr inbounds %struct.s1, %struct.s1* %0, i64 0, i32 0
  %1 = load %struct.s3*, %struct.s3** %snode94, align 8
  %2 = getelementptr inbounds %struct.s3, %struct.s3* %1, i32 0, i32 0
  %3 = load i32, i32* %2, align 4
  %tobool95.i = icmp eq i32 %3, 0
  br i1 %tobool95.i, label %if.then96.i, label %dup2

if.then96.i:                                      ; preds = %worklist
  call fastcc void @free_s3(%struct.s2* %p2, %struct.s3* %1) #1
  br label %dup2
}

; Function Attrs: nounwind
declare fastcc void @free_s3(%struct.s2*, %struct.s3*) unnamed_addr #0

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a57" "target-features"="+crc,+crypto,+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
