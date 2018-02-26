; RUN: llc -verify-machineinstrs < %s

target datalayout = "E-p:64:64"
target triple = "powerpc64-apple-darwin8"

define void @glArrayElement_CompExec() {
entry:
        %tmp3 = and i64 0, -8388609             ; <i64> [#uses=1]
        br label %cond_true24
cond_false:             ; preds = %cond_true24
        ret void
cond_true24:            ; preds = %cond_true24, %entry
        %indvar.ph = phi i32 [ 0, %entry ], [ %indvar.next, %cond_true24 ]              ; <i32> [#uses=1]
        %indvar = add i32 0, %indvar.ph         ; <i32> [#uses=2]
        %code.0 = trunc i32 %indvar to i8               ; <i8> [#uses=1]
        %tmp5 = add i8 %code.0, 16              ; <i8> [#uses=1]
        %shift.upgrd.1 = zext i8 %tmp5 to i64           ; <i64> [#uses=1]
        %tmp7 = lshr i64 %tmp3, %shift.upgrd.1          ; <i64> [#uses=1]
        %tmp7.upgrd.2 = trunc i64 %tmp7 to i32          ; <i32> [#uses=1]
        %tmp8 = and i32 %tmp7.upgrd.2, 1                ; <i32> [#uses=1]
        %tmp8.upgrd.3 = icmp eq i32 %tmp8, 0            ; <i1> [#uses=1]
        %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=1]
        br i1 %tmp8.upgrd.3, label %cond_false, label %cond_true24
}

