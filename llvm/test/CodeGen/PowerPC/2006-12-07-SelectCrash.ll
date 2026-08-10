; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; RUN: llc -verify-machineinstrs < %s
; REQUIRES: default_triple

@qsz.b = external global i1             ; <ptr> [#uses=1]

define fastcc void @qst() {
entry:
        br i1 true, label %cond_next71, label %cond_true
cond_true:              ; preds = %entry
        ret void
cond_next71:            ; preds = %entry
        %tmp73.b = load i1, ptr @qsz.b              ; <i1> [#uses=1]
        %ii.4.ph = select i1 %tmp73.b, i64 4, i64 0             ; <i64> [#uses=1]
        br label %bb139
bb82:           ; preds = %bb139
        ret void
bb139:          ; preds = %bb139, %cond_next71
        %exitcond89 = icmp eq i64 0, %ii.4.ph           ; <i1> [#uses=1]
        br i1 %exitcond89, label %bb82, label %bb139
}

