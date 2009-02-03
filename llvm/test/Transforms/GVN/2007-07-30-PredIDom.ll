; RUN: llvm-as < %s | opt -gvn | llvm-dis

	%"struct.Block::$_16" = type { i32 }
	%struct.Exp = type { %struct.Exp_*, i32, i32, i32, %struct.Exp*, %struct.Exp*, %"struct.Exp::$_10", %"struct.Block::$_16", %"struct.Exp::$_12" }
	%"struct.Exp::$_10" = type { %struct.Exp* }
	%"struct.Exp::$_12" = type { %struct.Exp** }
	%struct.Exp_ = type { i32, i32, i32, i32, %struct.Id* }
	%struct.Id = type { i8*, i32, i32, i32, %"struct.Id::$_13" }
	%"struct.Id::$_13" = type { double }

define i8* @_ZN3Exp8toStringEj(%struct.Exp* %this, i32 %nextpc) {
entry:
	switch i32 0, label %bb970 [
		 i32 1, label %bb
		 i32 2, label %bb39
		 i32 3, label %bb195
		 i32 4, label %bb270
		 i32 5, label %bb418
		 i32 6, label %bb633
		 i32 7, label %bb810
		 i32 8, label %bb882
		 i32 9, label %bb925
	]

bb:		; preds = %entry
	store i8* null, i8** null
	br label %return

bb39:		; preds = %entry
	br i1 false, label %cond_true, label %cond_false132

cond_true:		; preds = %bb39
	br i1 false, label %cond_true73, label %cond_false

cond_true73:		; preds = %cond_true
	br i1 false, label %cond_true108, label %cond_next

cond_true108:		; preds = %cond_true73
	br label %cond_next

cond_next:		; preds = %cond_true108, %cond_true73
	br label %cond_next131

cond_false:		; preds = %cond_true
	br label %cond_next131

cond_next131:		; preds = %cond_false, %cond_next
	br label %cond_next141

cond_false132:		; preds = %bb39
	br label %cond_next141

cond_next141:		; preds = %cond_false132, %cond_next131
	br i1 false, label %cond_true169, label %cond_false175

cond_true169:		; preds = %cond_next141
	br label %cond_next181

cond_false175:		; preds = %cond_next141
	br label %cond_next181

cond_next181:		; preds = %cond_false175, %cond_true169
	br i1 false, label %cond_true189, label %cond_next191

cond_true189:		; preds = %cond_next181
	br label %cond_next191

cond_next191:		; preds = %cond_true189, %cond_next181
	store i8* null, i8** null
	br label %return

bb195:		; preds = %entry
	br i1 false, label %cond_true248, label %cond_false250

cond_true248:		; preds = %bb195
	br label %cond_next252

cond_false250:		; preds = %bb195
	br label %cond_next252

cond_next252:		; preds = %cond_false250, %cond_true248
	br i1 false, label %cond_true265, label %cond_next267

cond_true265:		; preds = %cond_next252
	br label %cond_next267

cond_next267:		; preds = %cond_true265, %cond_next252
	store i8* null, i8** null
	br label %return

bb270:		; preds = %entry
	br i1 false, label %cond_true338, label %cond_false340

cond_true338:		; preds = %bb270
	br label %cond_next342

cond_false340:		; preds = %bb270
	br label %cond_next342

cond_next342:		; preds = %cond_false340, %cond_true338
	br i1 false, label %cond_true362, label %cond_false364

cond_true362:		; preds = %cond_next342
	br label %cond_next366

cond_false364:		; preds = %cond_next342
	br label %cond_next366

cond_next366:		; preds = %cond_false364, %cond_true362
	br i1 false, label %cond_true393, label %cond_next395

cond_true393:		; preds = %cond_next366
	br label %cond_next395

cond_next395:		; preds = %cond_true393, %cond_next366
	br i1 false, label %cond_true406, label %cond_next408

cond_true406:		; preds = %cond_next395
	br label %cond_next408

cond_next408:		; preds = %cond_true406, %cond_next395
	br i1 false, label %cond_true413, label %cond_next415

cond_true413:		; preds = %cond_next408
	br label %cond_next415

cond_next415:		; preds = %cond_true413, %cond_next408
	store i8* null, i8** null
	br label %return

bb418:		; preds = %entry
	br i1 false, label %cond_true512, label %cond_false514

cond_true512:		; preds = %bb418
	br label %cond_next516

cond_false514:		; preds = %bb418
	br label %cond_next516

cond_next516:		; preds = %cond_false514, %cond_true512
	br i1 false, label %cond_true536, label %cond_false538

cond_true536:		; preds = %cond_next516
	br label %cond_next540

cond_false538:		; preds = %cond_next516
	br label %cond_next540

cond_next540:		; preds = %cond_false538, %cond_true536
	br i1 false, label %cond_true560, label %cond_false562

cond_true560:		; preds = %cond_next540
	br label %cond_next564

cond_false562:		; preds = %cond_next540
	br label %cond_next564

cond_next564:		; preds = %cond_false562, %cond_true560
	br i1 false, label %cond_true597, label %cond_next599

cond_true597:		; preds = %cond_next564
	br label %cond_next599

cond_next599:		; preds = %cond_true597, %cond_next564
	br i1 false, label %cond_true614, label %cond_next616

cond_true614:		; preds = %cond_next599
	br label %cond_next616

cond_next616:		; preds = %cond_true614, %cond_next599
	br i1 false, label %cond_true621, label %cond_next623

cond_true621:		; preds = %cond_next616
	br label %cond_next623

cond_next623:		; preds = %cond_true621, %cond_next616
	br i1 false, label %cond_true628, label %cond_next630

cond_true628:		; preds = %cond_next623
	br label %cond_next630

cond_next630:		; preds = %cond_true628, %cond_next623
	store i8* null, i8** null
	br label %return

bb633:		; preds = %entry
	br i1 false, label %cond_true667, label %cond_next669

cond_true667:		; preds = %bb633
	br label %cond_next669

cond_next669:		; preds = %cond_true667, %bb633
	br i1 false, label %cond_true678, label %cond_next791

cond_true678:		; preds = %cond_next669
	br label %bb735

bb679:		; preds = %bb735
	br i1 false, label %cond_true729, label %cond_next731

cond_true729:		; preds = %bb679
	br label %cond_next731

cond_next731:		; preds = %cond_true729, %bb679
	br label %bb735

bb735:		; preds = %cond_next731, %cond_true678
	br i1 false, label %bb679, label %bb743

bb743:		; preds = %bb735
	br i1 false, label %cond_true788, label %cond_next790

cond_true788:		; preds = %bb743
	br label %cond_next790

cond_next790:		; preds = %cond_true788, %bb743
	br label %cond_next791

cond_next791:		; preds = %cond_next790, %cond_next669
	br i1 false, label %cond_true805, label %cond_next807

cond_true805:		; preds = %cond_next791
	br label %cond_next807

cond_next807:		; preds = %cond_true805, %cond_next791
	store i8* null, i8** null
	br label %return

bb810:		; preds = %entry
	br i1 false, label %cond_true870, label %cond_next872

cond_true870:		; preds = %bb810
	br label %cond_next872

cond_next872:		; preds = %cond_true870, %bb810
	br i1 false, label %cond_true877, label %cond_next879

cond_true877:		; preds = %cond_next872
	br label %cond_next879

cond_next879:		; preds = %cond_true877, %cond_next872
	store i8* null, i8** null
	br label %return

bb882:		; preds = %entry
	br i1 false, label %cond_true920, label %cond_next922

cond_true920:		; preds = %bb882
	br label %cond_next922

cond_next922:		; preds = %cond_true920, %bb882
	store i8* null, i8** null
	br label %return

bb925:		; preds = %entry
	br i1 false, label %cond_true965, label %cond_next967

cond_true965:		; preds = %bb925
	br label %cond_next967

cond_next967:		; preds = %cond_true965, %bb925
	store i8* null, i8** null
	br label %return

bb970:		; preds = %entry
	unreachable
		; No predecessors!
	store i8* null, i8** null
	br label %return

return:		; preds = %0, %cond_next967, %cond_next922, %cond_next879, %cond_next807, %cond_next630, %cond_next415, %cond_next267, %cond_next191, %bb
	%retval980 = load i8** null		; <i8*> [#uses=1]
	ret i8* %retval980
}
