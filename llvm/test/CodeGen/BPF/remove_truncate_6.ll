; RUN: llc < %s -mtriple=bpf -mcpu=v1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v1 -mattr=+alu32 -verify-machineinstrs | FileCheck --check-prefix=CHECK-32 %s
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
define dso_local void @cal1(ptr nocapture readonly %a, ptr nocapture %b, i32 %k) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp eq i32 %k, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = load i16, ptr %a, align 2
  %wide.trip.count = zext i32 %k to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %e.09 = phi i16 [ %0, %for.body.preheader ], [ %neg, %for.body ]
  %conv = zext i16 %e.09 to i64
  %arrayidx = getelementptr inbounds i64, ptr %b, i64 %indvars.iv
; CHECK: r{{[0-9]+}} &= 65535
; CHECK-32: r{{[0-9]+}} &= 65535
  store i64 %conv, ptr %arrayidx, align 8
  %neg = xor i16 %e.09, -1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nofree norecurse nounwind optsize
define dso_local void @cal2(ptr nocapture readonly %a, ptr nocapture %b, i32 %k) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp eq i32 %k, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = load i16, ptr %a, align 2
  %wide.trip.count = zext i32 %k to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %e.09 = phi i16 [ %0, %for.body.preheader ], [ %neg, %for.body ]
  %conv = zext i16 %e.09 to i32
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
; CHECK: r{{[0-9]+}} &= 65535
; CHECK-32: w{{[0-9]+}} &= 65535
  store i32 %conv, ptr %arrayidx, align 4
  %neg = xor i16 %e.09, -1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
