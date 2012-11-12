; RUN: llc -regalloc=greedy < %s | FileCheck %s

; LSR shouldn't introduce more induction variables than needed, increasing
; register pressure and therefore spilling. There is more room for improvement
; here.

; CHECK: sub sp, #{{40|32|28|24}}

; CHECK: %for.inc
; CHECK: ldr{{(.w)?}} r{{.*}}, [sp, #
; CHECK: ldr{{(.w)?}} r{{.*}}, [sp, #
; CHECK: add

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%struct.partition_entry = type { i32, i32, i64, i64 }

define i32 @partition_overlap_check(%struct.partition_entry* nocapture %part, i32 %num_entries) nounwind readonly optsize ssp {
entry:
  %cmp79 = icmp sgt i32 %num_entries, 0
  br i1 %cmp79, label %outer.loop, label %for.end72

outer.loop:                                 ; preds = %for.inc69, %entry
  %overlap.081 = phi i32 [ %overlap.4, %for.inc69 ], [ 0, %entry ]
  %0 = phi i32 [ %inc71, %for.inc69 ], [ 0, %entry ]
  %offset = getelementptr %struct.partition_entry* %part, i32 %0, i32 2
  %len = getelementptr %struct.partition_entry* %part, i32 %0, i32 3
  %tmp5 = load i64* %offset, align 4, !tbaa !0
  %tmp15 = load i64* %len, align 4, !tbaa !0
  %add = add nsw i64 %tmp15, %tmp5
  br label %inner.loop

inner.loop:                                       ; preds = %for.inc, %outer.loop
  %overlap.178 = phi i32 [ %overlap.081, %outer.loop ], [ %overlap.4, %for.inc ]
  %1 = phi i32 [ 0, %outer.loop ], [ %inc, %for.inc ]
  %cmp23 = icmp eq i32 %0, %1
  br i1 %cmp23, label %for.inc, label %if.end

if.end:                                           ; preds = %inner.loop
  %len39 = getelementptr %struct.partition_entry* %part, i32 %1, i32 3
  %offset28 = getelementptr %struct.partition_entry* %part, i32 %1, i32 2
  %tmp29 = load i64* %offset28, align 4, !tbaa !0
  %tmp40 = load i64* %len39, align 4, !tbaa !0
  %add41 = add nsw i64 %tmp40, %tmp29
  %cmp44 = icmp sge i64 %tmp29, %tmp5
  %cmp47 = icmp slt i64 %tmp29, %add
  %or.cond = and i1 %cmp44, %cmp47
  %overlap.2 = select i1 %or.cond, i32 1, i32 %overlap.178
  %cmp52 = icmp sle i64 %add41, %add
  %cmp56 = icmp sgt i64 %add41, %tmp5
  %or.cond74 = and i1 %cmp52, %cmp56
  %overlap.3 = select i1 %or.cond74, i32 1, i32 %overlap.2
  %cmp61 = icmp sgt i64 %tmp29, %tmp5
  %cmp65 = icmp slt i64 %add41, %add
  %or.cond75 = or i1 %cmp61, %cmp65
  br i1 %or.cond75, label %for.inc, label %if.then66

if.then66:                                        ; preds = %if.end
  br label %for.inc

for.inc:                                          ; preds = %if.end, %if.then66, %inner.loop
  %overlap.4 = phi i32 [ %overlap.178, %inner.loop ], [ 1, %if.then66 ], [ %overlap.3, %if.end ]
  %inc = add nsw i32 %1, 1
  %exitcond = icmp eq i32 %inc, %num_entries
  br i1 %exitcond, label %for.inc69, label %inner.loop

for.inc69:                                        ; preds = %for.inc
  %inc71 = add nsw i32 %0, 1
  %exitcond83 = icmp eq i32 %inc71, %num_entries
  br i1 %exitcond83, label %for.end72, label %outer.loop

for.end72:                                        ; preds = %for.inc69, %entry
  %overlap.0.lcssa = phi i32 [ 0, %entry ], [ %overlap.4, %for.inc69 ]
  ret i32 %overlap.0.lcssa
}

!0 = metadata !{metadata !"long long", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
