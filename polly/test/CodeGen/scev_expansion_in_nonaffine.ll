; RUN: opt %loadNPMPolly -passes=polly-codegen -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s

; bugpoint-reduced testcase of MiBench/consumer-lame/quantize-pvt.c from the
; test-suite.
; It features a SCEV that is used in two BasicBlock within a non-affine
; subregion where none of the blocks dominate the other. We check that the SCEV
; is expanded independently for both BasicBlocks instead of just once for the
; whole subregion.

; CHECK-LABEL:  polly.stmt.if.then.110:
; CHECK:          %[[R1_1:[0-9]*]] = mul nuw nsw i64 %polly.indvar[[R0_1:[0-9]*]], 120
; CHECK:          %scevgep[[R1_2:[0-9]*]] = getelementptr i8, ptr %scevgep{{[0-9]*}}, i64 %[[R1_1]]
; CHECK:          store i32 0, ptr %scevgep[[R1_2]], align 8

; CHECK-LABEL:  polly.stmt.if.else:
; CHECK:          %[[R2_1:[0-9]*]] = mul nuw nsw i64 %polly.indvar[[R0_1]], 120
; CHECK:          %scevgep[[R2_2:[0-9]*]] = getelementptr i8, ptr %scevgep{{[0-9]*}}, i64 %[[R2_1]]
; CHECK:          store i32 21, ptr %scevgep[[R2_2]], align 8

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.gr_info.4.59.136.224.290 = type { i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, [4 x i32] }
%struct.gr_info_ss.5.60.137.225.291 = type { %struct.gr_info.4.59.136.224.290 }
%struct.anon.6.61.138.226.292 = type { [2 x %struct.gr_info_ss.5.60.137.225.291] }
%struct.III_side_info_t.7.62.139.227.293 = type { i32, i32, i32, [2 x [4 x i32]], [2 x %struct.anon.6.61.138.226.292] }
%struct.lame_global_flags.3.58.135.223.289 = type { i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, i32, float, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, i32 }

@convert_mdct = external global i32, align 4
@reduce_sidechannel = external global i32, align 4

; Function Attrs: nounwind uwtable
define void @iteration_init(ptr %gfp, ptr %l3_side, ptr %l3_enc) #0 {
entry:
  %resvDrain = getelementptr inbounds %struct.III_side_info_t.7.62.139.227.293, ptr %l3_side, i64 0, i32 2
  store i32 0, ptr %resvDrain, align 8
  store i32 0, ptr @convert_mdct, align 4
  store i32 0, ptr @reduce_sidechannel, align 4
  %mode_gr = getelementptr inbounds %struct.lame_global_flags.3.58.135.223.289, ptr %gfp, i64 0, i32 45
  %0 = load i32, ptr %mode_gr, align 8
  %cmp95.145 = icmp sgt i32 %0, 0
  br i1 %cmp95.145, label %for.cond.98.preheader, label %for.cond.120.preheader

for.cond.98.preheader:                            ; preds = %for.inc.117, %entry
  %indvars.iv157 = phi i64 [ %indvars.iv.next158, %for.inc.117 ], [ 0, %entry ]
  %stereo = getelementptr inbounds %struct.lame_global_flags.3.58.135.223.289, ptr %gfp, i64 0, i32 46
  %1 = load i32, ptr %stereo, align 4
  %cmp99.143 = icmp sgt i32 %1, 0
  br i1 %cmp99.143, label %for.body.101, label %for.inc.117

for.cond.120.preheader:                           ; preds = %for.inc.117, %entry
  ret void

for.body.101:                                     ; preds = %for.inc.114, %for.cond.98.preheader
  %indvars.iv155 = phi i64 [ %indvars.iv.next156, %for.inc.114 ], [ 0, %for.cond.98.preheader ]
  %block_type = getelementptr inbounds %struct.III_side_info_t.7.62.139.227.293, ptr %l3_side, i64 0, i32 4, i64 %indvars.iv157, i32 0, i64 %indvars.iv155, i32 0, i32 6
  %2 = load i32, ptr %block_type, align 8
  %cmp108 = icmp eq i32 %2, 2
  %sfb_lmax = getelementptr inbounds %struct.III_side_info_t.7.62.139.227.293, ptr %l3_side, i64 0, i32 4, i64 %indvars.iv157, i32 0, i64 %indvars.iv155, i32 0, i32 16
  br i1 %cmp108, label %if.then.110, label %if.else

if.then.110:                                      ; preds = %for.body.101
  store i32 0, ptr %sfb_lmax, align 8
  %sfb_smax = getelementptr inbounds %struct.III_side_info_t.7.62.139.227.293, ptr %l3_side, i64 0, i32 4, i64 %indvars.iv157, i32 0, i64 %indvars.iv155, i32 0, i32 17
  store i32 0, ptr %sfb_smax, align 4
  br label %for.inc.114

if.else:                                          ; preds = %for.body.101
  store i32 21, ptr %sfb_lmax, align 8
  %sfb_smax112 = getelementptr inbounds %struct.III_side_info_t.7.62.139.227.293, ptr %l3_side, i64 0, i32 4, i64 %indvars.iv157, i32 0, i64 %indvars.iv155, i32 0, i32 17
  store i32 12, ptr %sfb_smax112, align 4
  br label %for.inc.114

for.inc.114:                                      ; preds = %if.else, %if.then.110
  %indvars.iv.next156 = add nuw nsw i64 %indvars.iv155, 1
  %3 = load i32, ptr %stereo, align 4
  %4 = sext i32 %3 to i64
  %cmp99 = icmp slt i64 %indvars.iv.next156, %4
  br i1 %cmp99, label %for.body.101, label %for.inc.117

for.inc.117:                                      ; preds = %for.inc.114, %for.cond.98.preheader
  %indvars.iv.next158 = add nuw nsw i64 %indvars.iv157, 1
  %5 = load i32, ptr %mode_gr, align 8
  %6 = sext i32 %5 to i64
  %cmp95 = icmp slt i64 %indvars.iv.next158, %6
  br i1 %cmp95, label %for.cond.98.preheader, label %for.cond.120.preheader
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
