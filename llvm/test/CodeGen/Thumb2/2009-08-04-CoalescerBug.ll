; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -relocation-model=pic -frame-pointer=all

	%0 = type { %struct.GAP }		; type %0
	%1 = type { i16, i8, i8 }		; type %1
	%2 = type { [2 x i32], [2 x i32] }		; type %2
	%3 = type { %struct.rec* }		; type %3
	%4 = type { i8, i8, i16, i8, i8, i8, i8 }		; type %4
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.FILE_POS = type { i8, i8, i16, i32 }
	%struct.FIRST_UNION = type { %struct.FILE_POS }
	%struct.FOURTH_UNION = type { %struct.STYLE }
	%struct.GAP = type { i8, i8, i16 }
	%struct.LIST = type { %struct.rec*, %struct.rec* }
	%struct.SECOND_UNION = type { %1 }
	%struct.STYLE = type { %0, %0, i16, i16, i32 }
	%struct.THIRD_UNION = type { %2 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.head_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, %struct.rec*, %3, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, i32 }
	%struct.rec = type { %struct.head_type }
@.str24239 = external constant [20 x i8], align 1		; <[20 x i8]*> [#uses=1]
@no_file_pos = external global %4		; <%4*> [#uses=1]
@zz_tmp = external global %struct.rec*		; <%struct.rec**> [#uses=1]
@.str81872 = external constant [10 x i8], align 1		; <[10 x i8]*> [#uses=1]
@out_fp = external global %struct.FILE*		; <%struct.FILE**> [#uses=2]
@cpexists = external global i32		; <i32*> [#uses=2]
@.str212784 = external constant [17 x i8], align 1		; <[17 x i8]*> [#uses=1]
@.str1822946 = external constant [8 x i8], align 1		; <[8 x i8]*> [#uses=1]
@.str1842948 = external constant [11 x i8], align 1		; <[11 x i8]*> [#uses=1]

declare i32 @fprintf(%struct.FILE* nocapture, i8* nocapture, ...) nounwind

declare i32 @"\01_fwrite"(i8*, i32, i32, i8*)

declare %struct.FILE* @OpenIncGraphicFile(i8*, i8 zeroext, %struct.rec** nocapture, %struct.FILE_POS*, i32* nocapture) nounwind

declare void @Error(i32, i32, i8*, i32, %struct.FILE_POS*, ...) nounwind

declare i8* @fgets(i8*, i32, %struct.FILE* nocapture) nounwind

define void @PS_PrintGraphicInclude(%struct.rec* %x, i32 %colmark, i32 %rowmark) nounwind {
entry:
	br label %bb5

bb5:		; preds = %bb5, %entry
	%.pn = phi %struct.rec* [ %y.0, %bb5 ], [ undef, %entry ]		; <%struct.rec*> [#uses=1]
	%y.0.in = getelementptr %struct.rec, %struct.rec* %.pn, i32 0, i32 0, i32 0, i32 1, i32 0		; <%struct.rec**> [#uses=1]
	%y.0 = load %struct.rec*, %struct.rec** %y.0.in		; <%struct.rec*> [#uses=2]
	br i1 undef, label %bb5, label %bb6

bb6:		; preds = %bb5
	%0 = call  %struct.FILE* @OpenIncGraphicFile(i8* undef, i8 zeroext 0, %struct.rec** undef, %struct.FILE_POS* null, i32* undef) nounwind		; <%struct.FILE*> [#uses=1]
	br i1 false, label %bb.i, label %FontHalfXHeight.exit

bb.i:		; preds = %bb6
	br label %FontHalfXHeight.exit

FontHalfXHeight.exit:		; preds = %bb.i, %bb6
	br i1 undef, label %bb.i1, label %FontSize.exit

bb.i1:		; preds = %FontHalfXHeight.exit
	br label %FontSize.exit

FontSize.exit:		; preds = %bb.i1, %FontHalfXHeight.exit
	%1 = load i32, i32* undef, align 4		; <i32> [#uses=1]
	%2 = icmp ult i32 0, undef		; <i1> [#uses=1]
	br i1 %2, label %bb.i5, label %FontName.exit

bb.i5:		; preds = %FontSize.exit
	call  void (i32, i32, i8*, i32, %struct.FILE_POS*, ...) @Error(i32 1, i32 2, i8* getelementptr ([20 x i8], [20 x i8]* @.str24239, i32 0, i32 0), i32 0, %struct.FILE_POS* bitcast (%4* @no_file_pos to %struct.FILE_POS*), i8* getelementptr ([10 x i8], [10 x i8]* @.str81872, i32 0, i32 0)) nounwind
	br label %FontName.exit

FontName.exit:		; preds = %bb.i5, %FontSize.exit
	%3 = call  i32 (%struct.FILE*, i8*, ...) @fprintf(%struct.FILE* undef, i8* getelementptr ([8 x i8], [8 x i8]* @.str1822946, i32 0, i32 0), i32 %1, i8* undef) nounwind		; <i32> [#uses=0]
	%4 = call  i32 @"\01_fwrite"(i8* getelementptr ([11 x i8], [11 x i8]* @.str1842948, i32 0, i32 0), i32 1, i32 10, i8* undef) nounwind		; <i32> [#uses=0]
	%5 = sub i32 %colmark, undef		; <i32> [#uses=1]
	%6 = sub i32 %rowmark, undef		; <i32> [#uses=1]
	%7 = load %struct.FILE*, %struct.FILE** @out_fp, align 4		; <%struct.FILE*> [#uses=1]
	%8 = call  i32 (%struct.FILE*, i8*, ...) @fprintf(%struct.FILE* %7, i8* getelementptr ([17 x i8], [17 x i8]* @.str212784, i32 0, i32 0), i32 %5, i32 %6) nounwind		; <i32> [#uses=0]
	store i32 0, i32* @cpexists, align 4
	%9 = getelementptr %struct.rec, %struct.rec* %y.0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%10 = load i32, i32* %9, align 4		; <i32> [#uses=1]
	%11 = sub i32 0, %10		; <i32> [#uses=1]
	%12 = load %struct.FILE*, %struct.FILE** @out_fp, align 4		; <%struct.FILE*> [#uses=1]
	%13 = call  i32 (%struct.FILE*, i8*, ...) @fprintf(%struct.FILE* %12, i8* getelementptr ([17 x i8], [17 x i8]* @.str212784, i32 0, i32 0), i32 undef, i32 %11) nounwind		; <i32> [#uses=0]
	store i32 0, i32* @cpexists, align 4
	br label %bb100.outer.outer

bb100.outer.outer:		; preds = %bb79.critedge, %bb1.i3, %FontName.exit
	%x_addr.0.ph.ph = phi %struct.rec* [ %x, %FontName.exit ], [ null, %bb79.critedge ], [ null, %bb1.i3 ]		; <%struct.rec*> [#uses=1]
	%14 = getelementptr %struct.rec, %struct.rec* %x_addr.0.ph.ph, i32 0, i32 0, i32 1, i32 0		; <%struct.FILE_POS*> [#uses=0]
	br label %bb100.outer

bb.i80:		; preds = %bb3.i85
	br i1 undef, label %bb2.i84, label %bb2.i51

bb2.i84:		; preds = %bb100.outer, %bb.i80
	br i1 undef, label %bb3.i77, label %bb3.i85

bb3.i85:		; preds = %bb2.i84
	br i1 false, label %StringBeginsWith.exit88, label %bb.i80

StringBeginsWith.exit88:		; preds = %bb3.i85
	br i1 undef, label %bb3.i77, label %bb2.i51

bb2.i.i68:		; preds = %bb3.i77
	br label %bb3.i77

bb3.i77:		; preds = %bb2.i.i68, %StringBeginsWith.exit88, %bb2.i84
	br i1 false, label %bb1.i58, label %bb2.i.i68

bb1.i58:		; preds = %bb3.i77
	unreachable

bb.i47:		; preds = %bb3.i52
	br i1 undef, label %bb2.i51, label %bb2.i.i15.critedge

bb2.i51:		; preds = %bb.i47, %StringBeginsWith.exit88, %bb.i80
	%15 = load i8, i8* undef, align 1		; <i8> [#uses=0]
	br i1 false, label %StringBeginsWith.exit55thread-split, label %bb3.i52

bb3.i52:		; preds = %bb2.i51
	br i1 false, label %StringBeginsWith.exit55, label %bb.i47

StringBeginsWith.exit55thread-split:		; preds = %bb2.i51
	br label %StringBeginsWith.exit55

StringBeginsWith.exit55:		; preds = %StringBeginsWith.exit55thread-split, %bb3.i52
	br label %bb2.i41

bb2.i41:		; preds = %bb2.i41, %StringBeginsWith.exit55
	br label %bb2.i41

bb2.i.i15.critedge:		; preds = %bb.i47
	%16 = call  i8* @fgets(i8* undef, i32 512, %struct.FILE* %0) nounwind		; <i8*> [#uses=0]
	%iftmp.560.0 = select i1 undef, i32 2, i32 0		; <i32> [#uses=1]
	br label %bb100.outer

bb2.i8:		; preds = %bb100.outer
	br i1 undef, label %bb1.i3, label %bb79.critedge

bb1.i3:		; preds = %bb2.i8
	br label %bb100.outer.outer

bb79.critedge:		; preds = %bb2.i8
	store %struct.rec* null, %struct.rec** @zz_tmp, align 4
	br label %bb100.outer.outer

bb100.outer:		; preds = %bb2.i.i15.critedge, %bb100.outer.outer
	%state.0.ph = phi i32 [ 0, %bb100.outer.outer ], [ %iftmp.560.0, %bb2.i.i15.critedge ]		; <i32> [#uses=1]
	%cond = icmp eq i32 %state.0.ph, 1		; <i1> [#uses=1]
	br i1 %cond, label %bb2.i8, label %bb2.i84
}
