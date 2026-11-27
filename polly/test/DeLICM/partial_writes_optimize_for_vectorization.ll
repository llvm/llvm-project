; RUN: opt %loadNPMPolly '-passes=polly-custom<prepare;delicm>' -plugin-arg=Polly,-polly-print-delicm -disable-output < %s | FileCheck %s --check-prefix=PARTIAL
; RUN: opt %loadNPMPolly '-passes=polly-custom<prepare;delicm>' -plugin-arg=Polly,-polly-print-delicm -plugin-arg=Polly,-polly-optimize-for-vectorization -disable-output < %s | FileCheck %s --check-prefix=NOPARTIAL
; RUN: opt %loadNPMPolly '-passes=polly-custom<prepare;delicm>' -plugin-arg=Polly,-polly-print-delicm -plugin-arg=Polly,-polly-optimize-for-vectorization -plugin-arg=Polly,-polly-delicm-partial-writes=true -disable-output < %s | FileCheck %s --check-prefix=PARTIAL
;
; Check that the partial PHI write mapping x to off[i+1] is not done with
; -polly-optimize-for-vectorization, unless -polly-delicm-partial-writes is
; passed explicitly.
;
;    void func(int n, int *off, int *base, int stride) {
;      int x = base[0];
;      for (int i = 0; i < n; i++) {
;        off[i] = x;
;        x += stride;
;      }
;    }

define void @func(i32 %n, ptr %off, ptr %base, i32 %stride) {
entry:
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %0 = load i32, ptr %base, align 4
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %x.06 = phi i32 [ %0, %for.body.preheader ], [ %add, %for.body ]
  %arrayidx1 = getelementptr inbounds nuw [4 x i8], ptr %off, i64 %indvars.iv
  store i32 %x.06, ptr %arrayidx1, align 4
  %add = add nsw i32 %x.06, %stride
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}


; PARTIAL:      After accesses {
; PARTIAL:          Stmt_for_body_preheader
; PARTIAL:             new: [n] -> { Stmt_for_body_preheader[] -> MemRef_off[0] };
; PARTIAL:          Stmt_for_body
; PARTIAL:             new: [n] -> { Stmt_for_body[i0] -> MemRef_off[1 + i0] : i0 <= -2 + n };
; PARTIAL:             new: [n] -> { Stmt_for_body[i0] -> MemRef_off[i0] };

; NOPARTIAL: No modification has been made
