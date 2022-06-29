; REQUIRES: asserts
; RUN: opt < %s -loop-interchange -cache-line-size=64 -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -pass-remarks-output=%t \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange -stats -S 2>&1 \
; RUN:     | FileCheck -check-prefix=STATS %s
; RUN: FileCheck -input-file %t %s


; no_deps_interchange just accesses a single nested array and can be interchange.
; CHECK:      Name:       Interchanged
; CHECK-NEXT: Function:   no_deps_interchange
define i32 @no_deps_interchange([1024 x i32]* nocapture %Arr) local_unnamed_addr #0 {
entry:
  br label %for1.header

for1.header:                                         ; preds = %entry, %for1.inc
  %indvars.iv19 = phi i64 [ 0, %entry ], [ %indvars.iv.next20, %for1.inc ]
  br label %for2

for2:                                        ; preds = %for1.header, %for2
  %indvars.iv = phi i64 [ 0, %for1.header ], [ %indvars.iv.next, %for2 ]
  %arrayidx6 = getelementptr inbounds [1024 x i32], [1024 x i32]* %Arr, i64 %indvars.iv, i64 %indvars.iv19
  store i32 0, i32* %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for2, label %for1.inc

for1.inc:
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond21 = icmp ne i64 %indvars.iv.next20, 1024
  br i1 %exitcond21, label %for1.header, label %exit

exit:                                 ; preds = %for1.inc
  ret i32 0

}

; No memory access using any induction variables, interchanging not beneficial.
; CHECK:      Name:        InterchangeNotProfitable
; CHECK-NEXT: Function:    no_mem_instrs
define i32 @no_mem_instrs(i64* %ptr) {
entry:
  br label %for1.header

for1.header:                                         ; preds = %entry, %for1.inc
  %indvars.iv19 = phi i64 [ 0, %entry ], [ %indvars.iv.next20, %for1.inc ]
  br label %for2

for2:                                        ; preds = %for1.header, %for2
  %indvars.iv = phi i64 [ 0, %for1.header ], [ %indvars.iv.next, %for2 ]
  store i64 %indvars.iv, i64* %ptr, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for2, label %for1.inc

for1.inc:
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond21 = icmp ne i64 %indvars.iv.next20, 1024
  br i1 %exitcond21, label %for1.header, label %exit

exit:                                 ; preds = %for1.inc
  ret i32 0
}


; Check stats, we interchanged 1 out of 3 loops.
; STATS: 1 loop-interchange - Number of loops interchanged
