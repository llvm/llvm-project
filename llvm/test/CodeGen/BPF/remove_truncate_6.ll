; RUN: llc < %s -march=bpf -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=bpf -mattr=+alu32 -verify-machineinstrs | FileCheck --check-prefix=CHECK-32 %s
;
; void cal1(unsigned short *a, unsigned long *b, unsigned int k)
; {
;   unsigned short e;
;
;   e = *a;
;   for (unsigned int i = 0; i < k; i++) {
;     b[i] = e;
;     e = ~e;
;   }
; }
;
; void cal2(unsigned short *a, unsigned int *b, unsigned int k)
; {
;   unsigned short e;
;
;   e = *a;
;   for (unsigned int i = 0; i < k; i++) {
;     b[i] = e;
;     e = ~e;
;   }
; }

; Function Attrs: nofree norecurse nounwind optsize
define dso_local void @cal1(i16* nocapture readonly %a, i64* nocapture %b, i32 %k) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp eq i32 %k, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = load i16, i16* %a, align 2
  %wide.trip.count = zext i32 %k to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %e.09 = phi i16 [ %0, %for.body.preheader ], [ %neg, %for.body ]
  %conv = zext i16 %e.09 to i64
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %indvars.iv
; CHECK: r{{[0-9]+}} &= 65535
; CHECK-32: r{{[0-9]+}} &= 65535
  store i64 %conv, i64* %arrayidx, align 8
  %neg = xor i16 %e.09, -1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nofree norecurse nounwind optsize
define dso_local void @cal2(i16* nocapture readonly %a, i32* nocapture %b, i32 %k) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp eq i32 %k, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = load i16, i16* %a, align 2
  %wide.trip.count = zext i32 %k to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %e.09 = phi i16 [ %0, %for.body.preheader ], [ %neg, %for.body ]
  %conv = zext i16 %e.09 to i32
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
; CHECK: r{{[0-9]+}} &= 65535
; CHECK-32: w{{[0-9]+}} &= 65535
  store i32 %conv, i32* %arrayidx, align 4
  %neg = xor i16 %e.09, -1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
