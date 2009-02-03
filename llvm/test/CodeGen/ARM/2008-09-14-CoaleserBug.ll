; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin

@"\01LC1" = external constant [288 x i8]		; <[288 x i8]*> [#uses=1]

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	br label %bb.i

bb.i:		; preds = %bb.i, %entry
	%i.01.i = phi i32 [ 0, %entry ], [ %indvar.next52, %bb.i ]		; <i32> [#uses=1]
	%indvar.next52 = add i32 %i.01.i, 1		; <i32> [#uses=2]
	%exitcond53 = icmp eq i32 %indvar.next52, 15		; <i1> [#uses=1]
	br i1 %exitcond53, label %bb.i33.loopexit, label %bb.i

bb.i33.loopexit:		; preds = %bb.i
	%0 = malloc [347 x i8]		; <[347 x i8]*> [#uses=2]
	%.sub = getelementptr [347 x i8]* %0, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %.sub, i8* getelementptr ([288 x i8]* @"\01LC1", i32 0, i32 0), i32 287, i32 1 ) nounwind
	br label %bb.i28

bb.i28:		; preds = %bb.i28, %bb.i33.loopexit
	br i1 false, label %repeat_fasta.exit, label %bb.i28

repeat_fasta.exit:		; preds = %bb.i28
	free [347 x i8]* %0
	unreachable
}
