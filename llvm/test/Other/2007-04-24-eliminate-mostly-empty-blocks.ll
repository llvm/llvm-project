;RUN: opt < %s -codegenprepare -S | FileCheck %s

;CHECK: define void @foo()
;CHECK-NEXT: entry:
;CHECK-NEXT:  ret void

;CHECK: cond_next475:
;CHECK-NEXT: br label %cond_next475


define void @foo() {
entry:
	br i1 false, label %cond_next31, label %cond_true

cond_true:		; preds = %entry
	br i1 false, label %cond_true19, label %cond_next31

cond_true19:		; preds = %cond_true
	br i1 false, label %bb510, label %cond_next31

cond_next31:		; preds = %cond_true19, %cond_true, %entry
	br i1 false, label %cond_true61, label %cond_next78

cond_true61:		; preds = %cond_next31
	br label %cond_next78

cond_next78:		; preds = %cond_true61, %cond_next31
	br i1 false, label %cond_true93, label %bb.preheader

cond_true93:		; preds = %cond_next78
	br label %bb.preheader

bb.preheader:		; preds = %cond_true93, %cond_next78
	%iftmp.11.0.ph.ph = phi i16 [ 0, %cond_true93 ], [ 0, %cond_next78 ]		; <i16> [#uses=1]
	br label %bb

bb:		; preds = %cond_next499, %bb.preheader
	%n.1 = phi i16 [ %iftmp.11.0.ph.ph, %cond_next499 ], [ 0, %bb.preheader ]		; <i16> [#uses=0]
	br i1 false, label %bb148.preheader, label %bb493

bb148.preheader:		; preds = %bb
	br label %bb148

bb148:		; preds = %cond_next475, %bb148.preheader
	br i1 false, label %cond_next175, label %bb184

cond_next175:		; preds = %bb148
	br i1 false, label %bb184, label %bb185

bb184:		; preds = %cond_next175, %bb148
	br label %bb185

bb185:		; preds = %bb184, %cond_next175
	br i1 false, label %bb420.preheader, label %cond_true198

bb420.preheader:		; preds = %bb185
	br label %bb420

cond_true198:		; preds = %bb185
	br i1 false, label %bb294, label %cond_next208

cond_next208:		; preds = %cond_true198
	br i1 false, label %cond_next249, label %cond_true214

cond_true214:		; preds = %cond_next208
	br i1 false, label %bb294, label %cond_next262

cond_next249:		; preds = %cond_next208
	br i1 false, label %bb294, label %cond_next262

cond_next262:		; preds = %cond_next249, %cond_true214
	br label %bb269

bb269:		; preds = %cond_next285, %cond_next262
	br i1 false, label %cond_next285, label %cond_true279

cond_true279:		; preds = %bb269
	br label %cond_next285

cond_next285:		; preds = %cond_true279, %bb269
	br i1 false, label %bb269, label %cond_next446.loopexit

bb294:		; preds = %cond_next249, %cond_true214, %cond_true198
	br i1 false, label %cond_next336, label %cond_true301

cond_true301:		; preds = %bb294
	br i1 false, label %cond_false398, label %cond_true344

cond_next336:		; preds = %bb294
	br i1 false, label %cond_false398, label %cond_true344

cond_true344:		; preds = %cond_next336, %cond_true301
	br i1 false, label %cond_false381, label %cond_true351

cond_true351:		; preds = %cond_true344
	br label %cond_next387

cond_false381:		; preds = %cond_true344
	br label %cond_next387

cond_next387:		; preds = %cond_false381, %cond_true351
	br label %cond_next401

cond_false398:		; preds = %cond_next336, %cond_true301
	br label %cond_next401

cond_next401:		; preds = %cond_false398, %cond_next387
	br i1 false, label %cond_next475, label %cond_true453

bb420:		; preds = %cond_next434, %bb420.preheader
	br i1 false, label %cond_next434, label %cond_true428

cond_true428:		; preds = %bb420
	br label %cond_next434

cond_next434:		; preds = %cond_true428, %bb420
	br i1 false, label %bb420, label %cond_next446.loopexit1

cond_next446.loopexit:		; preds = %cond_next285
	br label %cond_next446

cond_next446.loopexit1:		; preds = %cond_next434
	br label %cond_next446

cond_next446:		; preds = %cond_next446.loopexit1, %cond_next446.loopexit
	br i1 false, label %cond_next475, label %cond_true453

cond_true453:		; preds = %cond_next446, %cond_next401
	br i1 false, label %cond_true458, label %cond_next475

cond_true458:		; preds = %cond_true453
	br label %cond_next475

cond_next475:		; preds = %cond_true458, %cond_true453, %cond_next446, %cond_next401
	br i1 false, label %bb493.loopexit, label %bb148

bb493.loopexit:		; preds = %cond_next475
	br label %bb493

bb493:		; preds = %bb493.loopexit, %bb
	br i1 false, label %cond_next499, label %bb510.loopexit

cond_next499:		; preds = %bb493
	br label %bb

bb510.loopexit:		; preds = %bb493
	br label %bb510

bb510:		; preds = %bb510.loopexit, %cond_true19
	br i1 false, label %cond_next524, label %cond_true517

cond_true517:		; preds = %bb510
	br label %cond_next524

cond_next524:		; preds = %cond_true517, %bb510
	br i1 false, label %cond_next540, label %cond_true533

cond_true533:		; preds = %cond_next524
	br label %cond_next540

cond_next540:		; preds = %cond_true533, %cond_next524
	br i1 false, label %cond_true554, label %cond_next560

cond_true554:		; preds = %cond_next540
	br label %cond_next560

cond_next560:		; preds = %cond_true554, %cond_next540
	br i1 false, label %cond_true566, label %cond_next572

cond_true566:		; preds = %cond_next560
	br label %cond_next572

cond_next572:		; preds = %cond_true566, %cond_next560
	br i1 false, label %bb608.preheader, label %bb791.preheader

bb608.preheader:		; preds = %cond_next797.us, %cond_next572
	br label %bb608

bb608:		; preds = %cond_next771, %bb608.preheader
	br i1 false, label %cond_false627, label %cond_true613

cond_true613:		; preds = %bb608
	br label %cond_next640

cond_false627:		; preds = %bb608
	br label %cond_next640

cond_next640:		; preds = %cond_false627, %cond_true613
	br i1 false, label %cond_true653, label %cond_next671

cond_true653:		; preds = %cond_next640
	br label %cond_next671

cond_next671:		; preds = %cond_true653, %cond_next640
	br i1 false, label %cond_true683, label %cond_next724

cond_true683:		; preds = %cond_next671
	br i1 false, label %cond_next724, label %L1

cond_next724:		; preds = %cond_true683, %cond_next671
	br i1 false, label %cond_true735, label %L1

cond_true735:		; preds = %cond_next724
	br label %L1

L1:		; preds = %cond_true735, %cond_next724, %cond_true683
	br i1 false, label %cond_true745, label %cond_next771

cond_true745:		; preds = %L1
	br label %cond_next771

cond_next771:		; preds = %cond_true745, %L1
	br i1 false, label %bb608, label %bb791.preheader.loopexit

bb791.preheader.loopexit:		; preds = %cond_next771
	br label %bb791.preheader

bb791.preheader:		; preds = %bb791.preheader.loopexit, %cond_next572
	br i1 false, label %cond_next797.us, label %bb809.split

cond_next797.us:		; preds = %bb791.preheader
	br label %bb608.preheader

bb809.split:		; preds = %bb791.preheader
	br i1 false, label %cond_next827, label %cond_true820

cond_true820:		; preds = %bb809.split
	br label %cond_next827

cond_next827:		; preds = %cond_true820, %bb809.split
	br i1 false, label %cond_true833, label %cond_next840

cond_true833:		; preds = %cond_next827
	br label %cond_next840

cond_next840:		; preds = %cond_true833, %cond_next827
	br i1 false, label %bb866, label %bb1245

bb866:		; preds = %bb1239, %cond_next840
	br i1 false, label %cond_true875, label %bb911

cond_true875:		; preds = %bb866
	br label %cond_next1180

bb911:		; preds = %bb866
	switch i32 0, label %bb1165 [
		 i32 0, label %bb915
		 i32 1, label %bb932
		 i32 2, label %bb941
		 i32 3, label %bb1029
		 i32 4, label %bb1036
		 i32 5, label %bb1069
		 i32 6, label %L3
	]

bb915:		; preds = %cond_next1171, %bb911
	br i1 false, label %cond_next1171, label %cond_next1180

bb932:		; preds = %cond_next1171, %bb911
	br label %L1970

bb941:		; preds = %cond_next1171, %bb911
	br label %L1970

L1970:		; preds = %bb941, %bb932
	br label %bb1165

bb1029:		; preds = %cond_next1171, %bb911
	br label %L4

bb1036:		; preds = %cond_next1171, %bb911
	br label %L4

bb1069:		; preds = %cond_next1171, %bb911
	br i1 false, label %cond_next1121, label %cond_true1108

L3:		; preds = %cond_next1171, %bb911
	br i1 false, label %cond_next1121, label %cond_true1108

cond_true1108:		; preds = %L3, %bb1069
	br label %L4

cond_next1121:		; preds = %L3, %bb1069
	br label %L4

L4:		; preds = %cond_next1121, %cond_true1108, %bb1036, %bb1029
	br label %bb1165

bb1165:		; preds = %cond_next1171, %L4, %L1970, %bb911
	br i1 false, label %cond_next1171, label %cond_next1180

cond_next1171:		; preds = %bb1165, %bb915
	switch i32 0, label %bb1165 [
		 i32 0, label %bb915
		 i32 1, label %bb932
		 i32 2, label %bb941
		 i32 3, label %bb1029
		 i32 4, label %bb1036
		 i32 5, label %bb1069
		 i32 6, label %L3
	]

cond_next1180:		; preds = %bb1165, %bb915, %cond_true875
	br label %bb1239

bb1239:		; preds = %cond_next1251, %cond_next1180
	br i1 false, label %bb866, label %bb1245

bb1245:		; preds = %bb1239, %cond_next840
	br i1 false, label %cond_next1251, label %bb1257

cond_next1251:		; preds = %bb1245
	br label %bb1239

bb1257:		; preds = %bb1245
	ret void
}
