; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin 
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin -stats -info-output-file - | grep "Number of re-materialization" | grep 3

	%struct.CONTENTBOX = type { i32, i32, i32, i32, i32 }
	%struct.LOCBOX = type { i32, i32, i32, i32 }
	%struct.SIDEBOX = type { i32, i32 }
	%struct.UNCOMBOX = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.cellbox = type { i8*, i32, i32, i32, [9 x i32], i32, i32, i32, i32, i32, i32, i32, double, double, double, double, double, i32, i32, %struct.CONTENTBOX*, %struct.UNCOMBOX*, [8 x %struct.tilebox*], %struct.SIDEBOX* }
	%struct.termbox = type { %struct.termbox*, i32, i32, i32, i32, i32 }
	%struct.tilebox = type { %struct.tilebox*, double, double, double, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.termbox*, %struct.LOCBOX* }
@numcells = external global i32		; <i32*> [#uses=1]
@cellarray = external global %struct.cellbox**		; <%struct.cellbox***> [#uses=1]
@numBinsY = external global i32		; <i32*> [#uses=1]

define fastcc void @fixpenal() {
entry:
	%tmp491 = load i32* @numcells, align 4		; <i32> [#uses=1]
	%tmp9 = load %struct.cellbox*** @cellarray, align 4		; <%struct.cellbox**> [#uses=1]
	%tmp77.i = load i32* @numBinsY, align 4		; <i32> [#uses=2]
	br label %bb490

bb8:		; preds = %bb490, %cond_false428
  %foo3 = phi i1 [ 0, %bb490 ], [ 1, %cond_false428 ]
	br i1 %foo3, label %cond_false58.i, label %cond_false.i

cond_false.i:		; preds = %bb8
	ret void

cond_false58.i:		; preds = %bb8
	%highBinX.0.i = select i1 false, i32 1, i32 0		; <i32> [#uses=2]
	br i1 %foo3, label %cond_next85.i, label %cond_false76.i

cond_false76.i:		; preds = %cond_false58.i
	ret void

cond_next85.i:		; preds = %cond_false58.i
	br i1 %foo3, label %cond_next105.i, label %cond_false98.i

cond_false98.i:		; preds = %cond_next85.i
	ret void

cond_next105.i:		; preds = %cond_next85.i
	%tmp108.i = icmp eq i32 1, %highBinX.0.i		; <i1> [#uses=1]
	%tmp115.i = icmp eq i32 1, %tmp77.i		; <i1> [#uses=1]
	%bothcond.i = and i1 %tmp115.i, %tmp108.i		; <i1> [#uses=1]
	%storemerge.i = select i1 %bothcond.i, i32 1, i32 0		; <i32> [#uses=2]
	br i1 %bothcond.i, label %whoOverlaps.exit, label %bb503.preheader.i

bb503.preheader.i:		; preds = %bb513.i, %cond_next105.i
	%i.022.0.i = phi i32 [ %tmp512.i, %bb513.i ], [ 0, %cond_next105.i ]		; <i32> [#uses=2]
	%tmp165.i = getelementptr i32*** null, i32 %i.022.0.i		; <i32***> [#uses=0]
	br label %bb503.i

bb137.i:		; preds = %bb503.i
	br i1 %tmp506.i, label %bb162.i, label %bb148.i

bb148.i:		; preds = %bb137.i
	ret void

bb162.i:		; preds = %bb137.i
	%tmp49435.i = load i32* null		; <i32> [#uses=1]
	br label %bb170.i

bb170.i:		; preds = %bb491.i, %bb162.i
	%indvar.i = phi i32 [ %k.032.0.i, %bb491.i ], [ 0, %bb162.i ]		; <i32> [#uses=2]
	%k.032.0.i = add i32 %indvar.i, 1		; <i32> [#uses=2]
	%tmp173.i = getelementptr i32* null, i32 %k.032.0.i		; <i32*> [#uses=1]
	%tmp174.i = load i32* %tmp173.i		; <i32> [#uses=4]
	%tmp177.i = icmp eq i32 %tmp174.i, %cell.1		; <i1> [#uses=1]
	%tmp184.i = icmp sgt i32 %tmp174.i, %tmp491		; <i1> [#uses=1]
	%bothcond = or i1 %tmp177.i, %tmp184.i		; <i1> [#uses=1]
	br i1 %bothcond, label %bb491.i, label %cond_next188.i

cond_next188.i:		; preds = %bb170.i
	%tmp191.i = getelementptr %struct.cellbox** %tmp9, i32 %tmp174.i		; <%struct.cellbox**> [#uses=1]
	%tmp192.i = load %struct.cellbox** %tmp191.i		; <%struct.cellbox*> [#uses=1]
	%tmp195.i = icmp eq i32 %tmp174.i, 0		; <i1> [#uses=1]
	br i1 %tmp195.i, label %bb491.i, label %cond_true198.i

cond_true198.i:		; preds = %cond_next188.i
	%tmp210.i = getelementptr %struct.cellbox* %tmp192.i, i32 0, i32 3		; <i32*> [#uses=0]
	ret void

bb491.i:		; preds = %cond_next188.i, %bb170.i
	%tmp490.i = add i32 %indvar.i, 2		; <i32> [#uses=1]
	%tmp496.i = icmp slt i32 %tmp49435.i, %tmp490.i		; <i1> [#uses=1]
	br i1 %tmp496.i, label %bb500.i, label %bb170.i

bb500.i:		; preds = %bb491.i
	%indvar.next82.i = add i32 %j.0.i, 1		; <i32> [#uses=1]
	br label %bb503.i

bb503.i:		; preds = %bb500.i, %bb503.preheader.i
	%j.0.i = phi i32 [ 0, %bb503.preheader.i ], [ %indvar.next82.i, %bb500.i ]		; <i32> [#uses=2]
	%tmp506.i = icmp sgt i32 %j.0.i, %tmp77.i		; <i1> [#uses=1]
	br i1 %tmp506.i, label %bb513.i, label %bb137.i

bb513.i:		; preds = %bb503.i
	%tmp512.i = add i32 %i.022.0.i, 1		; <i32> [#uses=2]
	%tmp516.i = icmp sgt i32 %tmp512.i, %highBinX.0.i		; <i1> [#uses=1]
	br i1 %tmp516.i, label %whoOverlaps.exit, label %bb503.preheader.i

whoOverlaps.exit:		; preds = %bb513.i, %cond_next105.i
  %foo = phi i1 [ 1, %bb513.i], [0, %cond_next105.i]
	br i1 %foo, label %cond_false428, label %bb490

cond_false428:		; preds = %whoOverlaps.exit
	br i1 %foo, label %bb497, label %bb8

bb490:		; preds = %whoOverlaps.exit, %entry
	%binY.tmp.2 = phi i32 [ 0, %entry ], [ %storemerge.i, %whoOverlaps.exit ]		; <i32> [#uses=1]
	%cell.1 = phi i32 [ 1, %entry ], [ 0, %whoOverlaps.exit ]		; <i32> [#uses=1]
	%foo2 = phi i1 [ 1, %entry], [0, %whoOverlaps.exit]
	br i1 %foo2, label %bb497, label %bb8

bb497:		; preds = %bb490, %cond_false428
	%binY.tmp.3 = phi i32 [ %binY.tmp.2, %bb490 ], [ %storemerge.i, %cond_false428 ]		; <i32> [#uses=0]
	ret void
}
