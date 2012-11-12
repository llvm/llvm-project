; RUN: llc < %s -mtriple=arm-apple-darwin

@numBinsY = external global i32		; <i32*> [#uses=1]

declare double @pow(double, double)

define void @main(i32 %argc, i8** %argv) noreturn nounwind {
entry:
	br i1 false, label %bb34.outer.i.i.i, label %cond_false674
bb34.outer.i.i.i:		; preds = %entry
	br i1 false, label %bb2.i.i.i, label %bb47.i.i.i
bb2.i.i.i:		; preds = %bb34.outer.i.i.i
	%tmp24.i.i.i = call double @pow( double 0.000000e+00, double 2.000000e+00 )		; <double> [#uses=0]
	ret void
bb47.i.i.i:		; preds = %bb34.outer.i.i.i
	br i1 false, label %bb220.i.i.i, label %bb62.preheader.i.i.i
bb62.preheader.i.i.i:		; preds = %bb47.i.i.i
	ret void
bb220.i.i.i:		; preds = %bb47.i.i.i
	br i1 false, label %bb248.i.i.i, label %cond_next232.i.i.i
cond_next232.i.i.i:		; preds = %bb220.i.i.i
	ret void
bb248.i.i.i:		; preds = %bb220.i.i.i
	br i1 false, label %bb300.i.i.i, label %cond_false256.i.i.i
cond_false256.i.i.i:		; preds = %bb248.i.i.i
	ret void
bb300.i.i.i:		; preds = %bb248.i.i.i
	store i32 undef, i32* @numBinsY, align 4
	ret void
cond_false674:		; preds = %entry
	ret void
}

	%struct.anon = type { %struct.rnode*, %struct.rnode* }
	%struct.ch_set = type { { i8, i8 }*, %struct.ch_set* }
	%struct.pat_list = type { i32, %struct.pat_list* }
	%struct.rnode = type { i16, { %struct.anon }, i16, %struct.pat_list*, %struct.pat_list* }

define fastcc { i16, %struct.rnode* }* @get_token(i8** %s) nounwind  {
entry:
	br i1 false, label %bb42, label %bb78
bb42:		; preds = %entry
	br label %cond_next119.i
bb17.i:		; preds = %cond_next119.i
	br i1 false, label %cond_true53.i, label %cond_false99.i
cond_true53.i:		; preds = %bb17.i
	ret { i16, %struct.rnode* }* null
cond_false99.i:		; preds = %bb17.i
        %malloccall = tail call i8* @malloc(i32 trunc (i64 mul nuw (i64 ptrtoint (i1** getelementptr (i1** null, i32 1) to i64), i64 2) to i32))
        %tmp106.i = bitcast i8* %malloccall to %struct.ch_set*
	br i1 false, label %bb126.i, label %cond_next119.i
cond_next119.i:		; preds = %cond_false99.i, %bb42
	%curr_ptr.0.reg2mem.0.i = phi %struct.ch_set* [ %tmp106.i, %cond_false99.i ], [ null, %bb42 ]		; <%struct.ch_set*> [#uses=2]
	%prev_ptr.0.reg2mem.0.i = phi %struct.ch_set* [ %curr_ptr.0.reg2mem.0.i, %cond_false99.i ], [ undef, %bb42 ]		; <%struct.ch_set*> [#uses=1]
	br i1 false, label %bb126.i, label %bb17.i
bb126.i:		; preds = %cond_next119.i, %cond_false99.i
	%prev_ptr.0.reg2mem.1.i = phi %struct.ch_set* [ %prev_ptr.0.reg2mem.0.i, %cond_next119.i ], [ %curr_ptr.0.reg2mem.0.i, %cond_false99.i ]		; <%struct.ch_set*> [#uses=0]
	ret { i16, %struct.rnode* }* null
bb78:		; preds = %entry
	ret { i16, %struct.rnode* }* null
}

declare noalias i8* @malloc(i32)
