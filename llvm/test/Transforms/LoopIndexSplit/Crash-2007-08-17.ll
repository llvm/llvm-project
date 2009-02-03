; RUN: llvm-as < %s | opt -loop-index-split -disable-output 

        %struct._edit_script = type { %struct._edit_script*, i32, i8 }

define void @align_path(i8* %seq1, i8* %seq2, i32 %i1, i32 %j1, i32 %i2, i32 %j2, i32 %dist, %struct._edit_script** %head, %struct._edit_script** %tail, i32 %M, i32 %N) {
entry:
        br label %bb354

bb354:          ; preds = %bb511, %entry
        br i1 false, label %bb495, label %bb368

bb368:          ; preds = %bb354
        ret void

bb495:          ; preds = %bb495, %bb354
        br i1 false, label %bb511, label %bb495

bb511:          ; preds = %bb495
        br i1 false, label %xmalloc.exit69, label %bb354

xmalloc.exit69:         ; preds = %bb511
        br i1 false, label %bb556, label %bb542.preheader

bb542.preheader:                ; preds = %xmalloc.exit69
        ret void

bb556:          ; preds = %xmalloc.exit69
        br label %bb583

bb583:          ; preds = %cond_next693, %bb556
        %k.4342.0 = phi i32 [ %tmp707, %cond_next693 ], [ 0, %bb556 ]           ; <i32> [#uses=2]
        %tmp586 = icmp eq i32 %k.4342.0, 0              ; <i1> [#uses=1]
        br i1 %tmp586, label %cond_true589, label %cond_false608

cond_true589:           ; preds = %bb583
        br label %cond_next693

cond_false608:          ; preds = %bb583
        br i1 false, label %cond_next661, label %cond_next693

cond_next661:           ; preds = %cond_false608
        br label %cond_next693

cond_next693:           ; preds = %cond_next661, %cond_false608, %cond_true589
        %tmp705 = getelementptr i32* null, i32 0                ; <i32*> [#uses=0]
        %tmp707 = add i32 %k.4342.0, 1          ; <i32> [#uses=2]
        %tmp711 = icmp sgt i32 %tmp707, 0               ; <i1> [#uses=1]
        br i1 %tmp711, label %bb726.preheader, label %bb583

bb726.preheader:                ; preds = %cond_next693
        ret void
}
