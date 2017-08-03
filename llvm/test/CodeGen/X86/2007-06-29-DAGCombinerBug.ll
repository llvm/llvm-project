; RUN: llc < %s -march=x86 -mattr=+sse2

define void @test() {
entry:
	br i1 false, label %bb13944.preheader, label %cond_true418

cond_true418:		; preds = %entry
	ret void

bb13944.preheader:		; preds = %entry
	br i1 false, label %bb3517, label %bb13968.preheader

bb3517:		; preds = %bb13944.preheader
	br i1 false, label %cond_false7408, label %cond_next11422

cond_false7408:		; preds = %bb3517
	switch i32 0, label %cond_false10578 [
		 i32 7, label %cond_next11422
		 i32 6, label %cond_true7828
		 i32 1, label %cond_true10095
		 i32 3, label %cond_true10095
		 i32 5, label %cond_true10176
		 i32 24, label %cond_true10176
	]

cond_true7828:		; preds = %cond_false7408
	br i1 false, label %cond_next8191, label %cond_true8045

cond_true8045:		; preds = %cond_true7828
	ret void

cond_next8191:		; preds = %cond_true7828
	%tmp8234 = sub <4 x i32> < i32 939524096, i32 939524096, i32 939524096, i32 939524096 >, zeroinitializer		; <<4 x i32>> [#uses=0]
	ret void

cond_true10095:		; preds = %cond_false7408, %cond_false7408
	ret void

cond_true10176:		; preds = %cond_false7408, %cond_false7408
	ret void

cond_false10578:		; preds = %cond_false7408
	ret void

cond_next11422:		; preds = %cond_false7408, %bb3517
	ret void

bb13968.preheader:		; preds = %bb13944.preheader
	ret void
}
