; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu
; PR1766

        %struct.dentry = type { ptr }
        %struct.dentry_operations = type { ptr }
        %struct.qstr = type { i32, i32, ptr }

define ptr @d_hash_and_lookup(ptr %dir, ptr %name) {
entry:
        br i1 false, label %bb37, label %bb

bb:             ; preds = %bb, %entry
        %name8.0.reg2mem.0.rec = phi i64 [ %indvar.next, %bb ], [ 0, %entry ]           ; <i64> [#uses=1]
        %hash.0.reg2mem.0 = phi i64 [ %tmp27, %bb ], [ 0, %entry ]              ; <i64> [#uses=1]
        %tmp13 = load i8, ptr null, align 1         ; <i8> [#uses=1]
        %tmp1314 = zext i8 %tmp13 to i64                ; <i64> [#uses=1]
        %tmp25 = lshr i64 %tmp1314, 4           ; <i64> [#uses=1]
        %tmp22 = add i64 %tmp25, %hash.0.reg2mem.0              ; <i64> [#uses=1]
        %tmp26 = add i64 %tmp22, 0              ; <i64> [#uses=1]
        %tmp27 = mul i64 %tmp26, 11             ; <i64> [#uses=2]
        %indvar.next = add i64 %name8.0.reg2mem.0.rec, 1                ; <i64> [#uses=2]
        %exitcond = icmp eq i64 %indvar.next, 0         ; <i1> [#uses=1]
        br i1 %exitcond, label %bb37.loopexit, label %bb

bb37.loopexit:          ; preds = %bb
        %phitmp = trunc i64 %tmp27 to i32               ; <i32> [#uses=1]
        br label %bb37

bb37:           ; preds = %bb37.loopexit, %entry
        %hash.0.reg2mem.1 = phi i32 [ %phitmp, %bb37.loopexit ], [ 0, %entry ]          ; <i32> [#uses=1]
        store i32 %hash.0.reg2mem.1, ptr null, align 8
        %tmp75 = tail call i32 null( ptr %dir, ptr %name )                ; <i32> [#uses=0]
        %tmp84 = tail call i32 (...) @d_lookup( ptr %dir, ptr %name )            ; <i32> [#uses=0]
        ret ptr null
}

declare i32 @d_lookup(...)
