; RUN: llvm-as < %s | opt -disable-output -loop-unroll
; PR1770
; PR1947

	%struct.cl_engine = type { i32, i16, i32, i8**, i8**, i8*, i8*, i8*, i8*, i8*, i8*, i8* }
	%struct.cl_limits = type { i32, i32, i32, i32, i16, i64 }
	%struct.cli_ac_alt = type { i8, i8*, i16, i16, %struct.cli_ac_alt* }
	%struct.cli_ac_node = type { i8, i8, %struct.cli_ac_patt*, %struct.cli_ac_node**, %struct.cli_ac_node* }
	%struct.cli_ac_patt = type { i16*, i16*, i16, i16, i8, i32, i32, i8*, i8*, i32, i16, i16, i16, i16, %struct.cli_ac_alt**, i8, i16, %struct.cli_ac_patt*, %struct.cli_ac_patt* }
	%struct.cli_bm_patt = type { i8*, i32, i8*, i8*, i8, %struct.cli_bm_patt* }
	%struct.cli_ctx = type { i8**, i64*, %struct.cli_matcher*, %struct.cl_engine*, %struct.cl_limits*, i32, i32, i32, i32, %struct.cli_dconf* }
	%struct.cli_dconf = type { i32, i32, i32, i32, i32, i32, i32 }
	%struct.cli_matcher = type { i16, i8, i32*, %struct.cli_bm_patt**, i32*, i32, i8, i8, %struct.cli_ac_node*, %struct.cli_ac_node**, %struct.cli_ac_patt**, i32, i32, i32 }

declare i8* @calloc(i64, i64)

define fastcc i32 @cli_scanpe(i32 %desc, %struct.cli_ctx* %ctx) {
entry:
	br i1 false, label %cond_next17, label %cond_true14

cond_true14:		; preds = %entry
	ret i32 0

cond_next17:		; preds = %entry
	br i1 false, label %LeafBlock, label %LeafBlock1250

LeafBlock1250:		; preds = %cond_next17
	ret i32 0

LeafBlock:		; preds = %cond_next17
	br i1 false, label %cond_next33, label %cond_true30

cond_true30:		; preds = %LeafBlock
	ret i32 0

cond_next33:		; preds = %LeafBlock
	br i1 false, label %cond_next90, label %cond_true42

cond_true42:		; preds = %cond_next33
	ret i32 0

cond_next90:		; preds = %cond_next33
	br i1 false, label %cond_next100, label %cond_true97

cond_true97:		; preds = %cond_next90
	ret i32 0

cond_next100:		; preds = %cond_next90
	br i1 false, label %cond_next109, label %cond_true106

cond_true106:		; preds = %cond_next100
	ret i32 0

cond_next109:		; preds = %cond_next100
	br i1 false, label %cond_false, label %cond_true118

cond_true118:		; preds = %cond_next109
	ret i32 0

cond_false:		; preds = %cond_next109
	br i1 false, label %NodeBlock1482, label %cond_true126

cond_true126:		; preds = %cond_false
	ret i32 0

NodeBlock1482:		; preds = %cond_false
	br i1 false, label %cond_next285, label %NodeBlock1480

NodeBlock1480:		; preds = %NodeBlock1482
	ret i32 0

cond_next285:		; preds = %NodeBlock1482
	br i1 false, label %cond_next320, label %cond_true294

cond_true294:		; preds = %cond_next285
	ret i32 0

cond_next320:		; preds = %cond_next285
	br i1 false, label %LeafBlock1491, label %LeafBlock1493

LeafBlock1493:		; preds = %cond_next320
	ret i32 0

LeafBlock1491:		; preds = %cond_next320
	br i1 false, label %cond_true400, label %cond_true378

cond_true378:		; preds = %LeafBlock1491
	ret i32 1

cond_true400:		; preds = %LeafBlock1491
	br i1 false, label %cond_next413, label %cond_true406

cond_true406:		; preds = %cond_true400
	ret i32 0

cond_next413:		; preds = %cond_true400
	br i1 false, label %cond_next429, label %cond_true424

cond_true424:		; preds = %cond_next413
	ret i32 0

cond_next429:		; preds = %cond_next413
	br i1 false, label %NodeBlock1557, label %NodeBlock1579

NodeBlock1579:		; preds = %cond_next429
	ret i32 0

NodeBlock1557:		; preds = %cond_next429
	br i1 false, label %LeafBlock1543, label %NodeBlock1555

NodeBlock1555:		; preds = %NodeBlock1557
	ret i32 0

LeafBlock1543:		; preds = %NodeBlock1557
	br i1 false, label %cond_next870, label %cond_next663

cond_next663:		; preds = %LeafBlock1543
	ret i32 0

cond_next870:		; preds = %LeafBlock1543
	br i1 false, label %cond_true1012, label %cond_true916

cond_true916:		; preds = %cond_next870
	ret i32 0

cond_true1012:		; preds = %cond_next870
	br i1 false, label %cond_next3849, label %cond_true2105

cond_true2105:		; preds = %cond_true1012
	ret i32 0

cond_next3849:		; preds = %cond_true1012
	br i1 false, label %cond_next4378, label %bb6559

bb3862:		; preds = %cond_next4385
	br i1 false, label %cond_false3904, label %cond_true3876

cond_true3876:		; preds = %bb3862
	ret i32 0

cond_false3904:		; preds = %bb3862
	br i1 false, label %cond_next4003, label %cond_true3935

cond_true3935:		; preds = %cond_false3904
	ret i32 0

cond_next4003:		; preds = %cond_false3904
	br i1 false, label %cond_next5160, label %cond_next4015

cond_next4015:		; preds = %cond_next4003
	ret i32 0

cond_next4378:		; preds = %cond_next3849
	br i1 false, label %cond_next4385, label %bb4393

cond_next4385:		; preds = %cond_next4378
	br i1 false, label %bb3862, label %bb4393

bb4393:		; preds = %cond_next4385, %cond_next4378
	ret i32 0

cond_next5160:		; preds = %cond_next4003
	br i1 false, label %bb5188, label %bb6559

bb5188:		; preds = %cond_next5160
	br i1 false, label %cond_next5285, label %cond_true5210

cond_true5210:		; preds = %bb5188
	ret i32 0

cond_next5285:		; preds = %bb5188
	br i1 false, label %cond_true5302, label %cond_true5330

cond_true5302:		; preds = %cond_next5285
	br i1 false, label %bb7405, label %bb7367

cond_true5330:		; preds = %cond_next5285
	ret i32 0

bb6559:		; preds = %cond_next5160, %cond_next3849
	ret i32 0

bb7367:		; preds = %cond_true5302
	ret i32 0

bb7405:		; preds = %cond_true5302
	br i1 false, label %cond_next8154, label %cond_true7410

cond_true7410:		; preds = %bb7405
	ret i32 0

cond_next8154:		; preds = %bb7405
	br i1 false, label %cond_true8235, label %bb9065

cond_true8235:		; preds = %cond_next8154
	br i1 false, label %bb8274, label %bb8245

bb8245:		; preds = %cond_true8235
	ret i32 0

bb8274:		; preds = %cond_true8235
	br i1 false, label %cond_next8358, label %cond_true8295

cond_true8295:		; preds = %bb8274
	ret i32 0

cond_next8358:		; preds = %bb8274
	br i1 false, label %cond_next.i509, label %cond_true8371

cond_true8371:		; preds = %cond_next8358
	ret i32 -123

cond_next.i509:		; preds = %cond_next8358
	br i1 false, label %bb36.i, label %bb33.i

bb33.i:		; preds = %cond_next.i509
	ret i32 0

bb36.i:		; preds = %cond_next.i509
	br i1 false, label %cond_next54.i, label %cond_true51.i

cond_true51.i:		; preds = %bb36.i
	ret i32 0

cond_next54.i:		; preds = %bb36.i
	%tmp10.i.i527 = call i8* @calloc( i64 0, i64 1 )		; <i8*> [#uses=1]
	br i1 false, label %cond_next11.i.i, label %bb132.i

bb132.i:		; preds = %cond_next54.i
	ret i32 0

cond_next11.i.i:		; preds = %cond_next54.i
	br i1 false, label %bb32.i.i545, label %cond_true1008.critedge.i

bb32.i.i545:		; preds = %cond_next11.i.i
	br i1 false, label %cond_next349.i, label %cond_true184.i

cond_true184.i:		; preds = %bb32.i.i545
	ret i32 0

cond_next349.i:		; preds = %bb32.i.i545
	br i1 false, label %cond_next535.i, label %cond_true1008.critedge1171.i

cond_next535.i:		; preds = %cond_next349.i
	br i1 false, label %cond_next569.i, label %cond_false574.i

cond_next569.i:		; preds = %cond_next535.i
	br i1 false, label %cond_next670.i, label %cond_true1008.critedge1185.i

cond_false574.i:		; preds = %cond_next535.i
	ret i32 0

cond_next670.i:		; preds = %cond_next569.i
	br i1 false, label %cond_true692.i, label %cond_next862.i

cond_true692.i:		; preds = %cond_next670.i
	br i1 false, label %cond_false742.i, label %cond_true718.i

cond_true718.i:		; preds = %cond_true692.i
	ret i32 0

cond_false742.i:		; preds = %cond_true692.i
	br i1 false, label %cond_true784.i, label %cond_next9079

cond_true784.i:		; preds = %cond_next811.i, %cond_false742.i
	%indvar1411.i.reg2mem.0 = phi i8 [ %indvar.next1412.i, %cond_next811.i ], [ 0, %cond_false742.i ]		; <i8> [#uses=1]
	br i1 false, label %cond_true1008.critedge1190.i, label %cond_next811.i

cond_next811.i:		; preds = %cond_true784.i
	%indvar.next1412.i = add i8 %indvar1411.i.reg2mem.0, 1		; <i8> [#uses=2]
	%tmp781.i = icmp eq i8 %indvar.next1412.i, 3		; <i1> [#uses=1]
	br i1 %tmp781.i, label %cond_next9079, label %cond_true784.i

cond_next862.i:		; preds = %cond_next670.i
	ret i32 0

cond_true1008.critedge.i:		; preds = %cond_next11.i.i
	ret i32 0

cond_true1008.critedge1171.i:		; preds = %cond_next349.i
	ret i32 0

cond_true1008.critedge1185.i:		; preds = %cond_next569.i
	ret i32 0

cond_true1008.critedge1190.i:		; preds = %cond_true784.i
	%tmp621.i532.lcssa610 = phi i8* [ %tmp10.i.i527, %cond_true784.i ]		; <i8*> [#uses=0]
	ret i32 0

bb9065:		; preds = %cond_next8154
	ret i32 0

cond_next9079:		; preds = %cond_next811.i, %cond_false742.i
	ret i32 0
}
