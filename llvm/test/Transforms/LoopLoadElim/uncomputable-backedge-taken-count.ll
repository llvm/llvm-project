; RUN: opt -loop-load-elim -S < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes=loop-load-elim -S < %s | FileCheck %s

target datalayout = "e-m:o-i32:64-f80:128-n8:16:32:64-S128"

; TODO
; Make sure loop-load-elimination triggers for a loop with uncomputable
; backedge-taken counts when no runtime checks are required.
define void @load_elim_no_runtime_checks(i32* noalias %A, i32* noalias %B, i32* noalias %C, i32 %N) {
; CHECK-LABEL: load_elim_no_runtime_checks
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %for.body
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, i32* %A, i32 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, i32* %B, i32 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i32 %indvars.iv
  %Aidx = getelementptr inbounds i32, i32* %A, i32 %indvars.iv

  %b = load i32, i32* %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, i32* %Aidx_next, align 4

  %a = load i32, i32* %Aidx, align 1
  %c = mul i32 %a, 2
  store i32 %c, i32* %Cidx, align 4

  %exitcond = icmp eq i32 %indvars.iv.next, %a
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Make sure loop-load-elimination triggers for a loop with uncomputable
; backedge-taken counts when no runtime checks are required.
define void @load_elim_wrapping_runtime_checks(i32* noalias %A, i32* noalias %B, i32* noalias %C, i32 %N) {
; CHECK-LABEL: @load_elim_wrapping_runtime_checks
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %for.body
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i32 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, i32* %A, i32 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, i32* %B, i32 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i32 %indvars.iv
  %Aidx = getelementptr inbounds i32, i32* %A, i32 %indvars.iv

  %b = load i32, i32* %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, i32* %Aidx_next, align 4

  %a = load i32, i32* %Aidx, align 1
  %c = mul i32 %a, 2
  store i32 %c, i32* %Cidx, align 4

  %exitcond = icmp eq i32 %indvars.iv.next, %a
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Make sure we do not crash when dealing with uncomputable backedge-taken counts
; and a variable distance between accesses.
define void @uncomputable_btc_crash(i8* %row, i32 %filter, i32* noalias %exits) local_unnamed_addr #0 {
; CHECK-LABEL: @uncomputable_btc_crash
; CHECK-NEXT:  entry:
; CHECK-NEXT:    getelementptr
; CHECK-NEXT:    br label %loop
;
entry:
  %add.ptr = getelementptr inbounds i8, i8* %row, i32 %filter
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %add.ptr.gep = getelementptr i8, i8* %add.ptr, i32 %iv
  %row.gep = getelementptr i8, i8* %row, i32 %iv
  %gep.1 = getelementptr i8, i8* %add.ptr.gep, i32 0
  %gep.2 = getelementptr i8, i8* %row.gep, i32 0
  %l = load i8, i8* %gep.2, align 1
  store i8 %l, i8* %gep.1, align 1
  %iv.next = add i32 %iv, 8
  %exit.gep = getelementptr i32, i32* %exits, i32 %iv
  %lv = load i32, i32* %exit.gep
  %c = icmp eq i32 %lv, 120
  br i1 %c, label %exit, label %loop

exit:
  ret void
}
