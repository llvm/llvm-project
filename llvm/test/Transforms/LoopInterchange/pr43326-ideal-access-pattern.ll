; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa -stats 2>&1
; RUN: FileCheck --input-file=%t --check-prefix=REMARKS %s

; Triply nested loop, should be able to do interchange three times
; to get the ideal access pattern.
; void f(int e[10][10][10], int f[10][10][10]) {
;   for (int a = 0; a < 10; a++) {
;     for (int b = 0; b < 10; b++) {
;       for (int c = 0; c < 10; c++) {
;         f[c][b][a] = e[c][b][a];
;       }
;     }
;   }
; }

; REMARKS: --- !Passed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            Interchanged
; REMARKS-NEXT: Function:        pr43326-triply-nested
; REMARKS: --- !Passed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            Interchanged
; REMARKS-NEXT: Function:        pr43326-triply-nested
; REMARKS: --- !Passed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            Interchanged
; REMARKS-NEXT: Function:        pr43326-triply-nested

define void @pr43326-triply-nested(ptr %e, ptr %f) {
entry:
  br label %for.outermost.header

for.outermost.header:                              ; preds = %entry, %for.outermost.latch
  %indvars.outermost = phi i64 [ 0, %entry ], [ %indvars.outermost.next, %for.outermost.latch ]
  br label %for.middle.header

for.cond.cleanup:                                 ; preds = %for.outermost.latch
  ret void

for.middle.header:                              ; preds = %for.outermost.header, %for.middle.latch
  %indvars.middle = phi i64 [ 0, %for.outermost.header ], [ %indvars.middle.next, %for.middle.latch ]
  br label %for.innermost

for.outermost.latch:                                ; preds = %for.middle.latch
  %indvars.outermost.next = add nuw nsw i64 %indvars.outermost, 1
  %exitcond.outermost = icmp ne i64 %indvars.outermost.next, 10
  br i1 %exitcond.outermost, label %for.outermost.header, label %for.cond.cleanup

for.middle.latch:                                ; preds = %for.innermost
  %indvars.middle.next = add nuw nsw i64 %indvars.middle, 1
  %exitcond.middle = icmp ne i64 %indvars.middle.next, 10
  br i1 %exitcond.middle, label %for.middle.header, label %for.outermost.latch

for.innermost:                                        ; preds = %for.middle.header, %for.innermost
  %indvars.innermost = phi i64 [ 0, %for.middle.header ], [ %indvars.innermost.next, %for.innermost ]
  %arrayidx12 = getelementptr inbounds [10 x [10 x i32]], ptr %e, i64 %indvars.innermost, i64 %indvars.middle, i64 %indvars.outermost
  %0 = load i32, ptr %arrayidx12
  %arrayidx18 = getelementptr inbounds [10 x [10 x i32]], ptr %f, i64 %indvars.innermost, i64 %indvars.middle, i64 %indvars.outermost
  store i32 %0, ptr %arrayidx18
  %indvars.innermost.next = add nuw nsw i64 %indvars.innermost, 1
  %exitcond.innermost = icmp ne i64 %indvars.innermost.next, 10
  br i1 %exitcond.innermost, label %for.innermost, label %for.middle.latch
}
