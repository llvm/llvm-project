; REQUIRES: asserts
; RUN: opt -enable-shuffle-padding=true -enable-masked-interleaved-mem-accesses=true -passes=loop-vectorize -debug-only=loop-vectorize  -mtriple=aarch64 -mattr=+sve -aarch64-sve-vector-bits-min=512 -S < %s 2>&1  | FileCheck %s --check-prefixes=PADDING
; RUN: opt -enable-shuffle-padding=false -enable-masked-interleaved-mem-accesses=true -passes=loop-vectorize -debug-only=loop-vectorize  -mtriple=aarch64 -mattr=+sve -aarch64-sve-vector-bits-min=512 -S < %s 2>&1  | FileCheck %s --check-prefixes=NO-PADDING

%struct.patic = type { float, float, float }

; for (int i = 0; i < num; i++) {
;   ps[i].x = factor * ps[i].x;
;   ps[i].y = factor * ps[i].y;
; }
;
define void @shufflePadding(i32 noundef %num, ptr nocapture noundef %ps) {
; PADDING-LABEL: 'shufflePadding'
; PADDING: LV: Found an estimated cost of 3 for VF 16 For instruction:   store float %mul6, ptr %y, align 4

; NO-PADDING-LABEL: 'shufflePadding'
; NO-PADDING: LV: Found an estimated cost of 188 for VF 16 For instruction:   store float %mul6, ptr %y, align 4
entry:
  %cmp19 = icmp sgt i32 %num, 0
  br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %num to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds %struct.patic, ptr %ps, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul fast float %0, 0x40019999A0000000
  store float %mul, ptr %arrayidx, align 4
  %y = getelementptr inbounds %struct.patic, ptr %arrayidx, i64 0, i32 1
  %1 = load float, ptr %y, align 4
  %mul6 = fmul fast float %1, 0x40019999A0000000
  store float %mul6, ptr %y, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

