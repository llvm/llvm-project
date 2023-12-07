; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -relocation-model=pic -frame-pointer=all

	%0 = type { %struct.GAP }		; type %0
	%1 = type { i16, i8, i8 }		; type %1
	%2 = type { [2 x i32], [2 x i32] }		; type %2
	%3 = type { ptr }		; type %3
	%4 = type { i8, i8, i16, i8, i8, i8, i8 }		; type %4
	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.FILE_POS = type { i8, i8, i16, i32 }
	%struct.FIRST_UNION = type { %struct.FILE_POS }
	%struct.FOURTH_UNION = type { %struct.STYLE }
	%struct.GAP = type { i8, i8, i16 }
	%struct.LIST = type { ptr, ptr }
	%struct.SECOND_UNION = type { %1 }
	%struct.STYLE = type { %0, %0, i16, i16, i32 }
	%struct.THIRD_UNION = type { %2 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
	%struct.head_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, ptr, %3, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32 }
	%struct.rec = type { %struct.head_type }
@.str24239 = external constant [20 x i8], align 1		; <ptr> [#uses=1]
@no_file_pos = external global %4		; <ptr> [#uses=1]
@zz_tmp = external global ptr		; <ptr> [#uses=1]
@.str81872 = external constant [10 x i8], align 1		; <ptr> [#uses=1]
@out_fp = external global ptr		; <ptr> [#uses=2]
@cpexists = external global i32		; <ptr> [#uses=2]
@.str212784 = external constant [17 x i8], align 1		; <ptr> [#uses=1]
@.str1822946 = external constant [8 x i8], align 1		; <ptr> [#uses=1]
@.str1842948 = external constant [11 x i8], align 1		; <ptr> [#uses=1]

declare i32 @fprintf(ptr nocapture, ptr nocapture, ...) nounwind

declare i32 @"\01_fwrite"(ptr, i32, i32, ptr)

declare ptr @OpenIncGraphicFile(ptr, i8 zeroext, ptr nocapture, ptr, ptr nocapture) nounwind

declare void @Error(i32, i32, ptr, i32, ptr, ...) nounwind

declare ptr @fgets(ptr, i32, ptr nocapture) nounwind

define void @PS_PrintGraphicInclude(ptr %x, i32 %colmark, i32 %rowmark) nounwind {
entry:
	br label %bb5

bb5:		; preds = %bb5, %entry
	%.pn = phi ptr [ %y.0, %bb5 ], [ undef, %entry ]		; <ptr> [#uses=1]
	%y.0.in = getelementptr %struct.rec, ptr %.pn, i32 0, i32 0, i32 0, i32 1, i32 0		; <ptr> [#uses=1]
	%y.0 = load ptr, ptr %y.0.in		; <ptr> [#uses=2]
	br i1 undef, label %bb5, label %bb6

bb6:		; preds = %bb5
	%0 = call  ptr @OpenIncGraphicFile(ptr undef, i8 zeroext 0, ptr undef, ptr null, ptr undef) nounwind		; <ptr> [#uses=1]
	br i1 false, label %bb.i, label %FontHalfXHeight.exit

bb.i:		; preds = %bb6
	br label %FontHalfXHeight.exit

FontHalfXHeight.exit:		; preds = %bb.i, %bb6
	br i1 undef, label %bb.i1, label %FontSize.exit

bb.i1:		; preds = %FontHalfXHeight.exit
	br label %FontSize.exit

FontSize.exit:		; preds = %bb.i1, %FontHalfXHeight.exit
	%1 = load i32, ptr undef, align 4		; <i32> [#uses=1]
	%2 = icmp ult i32 0, undef		; <i1> [#uses=1]
	br i1 %2, label %bb.i5, label %FontName.exit

bb.i5:		; preds = %FontSize.exit
	call  void (i32, i32, ptr, i32, ptr, ...) @Error(i32 1, i32 2, ptr @.str24239, i32 0, ptr @no_file_pos, ptr @.str81872) nounwind
	br label %FontName.exit

FontName.exit:		; preds = %bb.i5, %FontSize.exit
	%3 = call  i32 (ptr, ptr, ...) @fprintf(ptr undef, ptr @.str1822946, i32 %1, ptr undef) nounwind		; <i32> [#uses=0]
	%4 = call  i32 @"\01_fwrite"(ptr @.str1842948, i32 1, i32 10, ptr undef) nounwind		; <i32> [#uses=0]
	%5 = sub i32 %colmark, undef		; <i32> [#uses=1]
	%6 = sub i32 %rowmark, undef		; <i32> [#uses=1]
	%7 = load ptr, ptr @out_fp, align 4		; <ptr> [#uses=1]
	%8 = call  i32 (ptr, ptr, ...) @fprintf(ptr %7, ptr @.str212784, i32 %5, i32 %6) nounwind		; <i32> [#uses=0]
	store i32 0, ptr @cpexists, align 4
	%9 = getelementptr %struct.rec, ptr %y.0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 1		; <ptr> [#uses=1]
	%10 = load i32, ptr %9, align 4		; <i32> [#uses=1]
	%11 = sub i32 0, %10		; <i32> [#uses=1]
	%12 = load ptr, ptr @out_fp, align 4		; <ptr> [#uses=1]
	%13 = call  i32 (ptr, ptr, ...) @fprintf(ptr %12, ptr @.str212784, i32 undef, i32 %11) nounwind		; <i32> [#uses=0]
	store i32 0, ptr @cpexists, align 4
	br label %bb100.outer.outer

bb100.outer.outer:		; preds = %bb79.critedge, %bb1.i3, %FontName.exit
	%x_addr.0.ph.ph = phi ptr [ %x, %FontName.exit ], [ null, %bb79.critedge ], [ null, %bb1.i3 ]		; <ptr> [#uses=1]
	%14 = getelementptr %struct.rec, ptr %x_addr.0.ph.ph, i32 0, i32 0, i32 1, i32 0		; <ptr> [#uses=0]
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
	%15 = load i8, ptr undef, align 1		; <i8> [#uses=0]
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
	%16 = call  ptr @fgets(ptr undef, i32 512, ptr %0) nounwind		; <ptr> [#uses=0]
	%iftmp.560.0 = select i1 undef, i32 2, i32 0		; <i32> [#uses=1]
	br label %bb100.outer

bb2.i8:		; preds = %bb100.outer
	br i1 undef, label %bb1.i3, label %bb79.critedge

bb1.i3:		; preds = %bb2.i8
	br label %bb100.outer.outer

bb79.critedge:		; preds = %bb2.i8
	store ptr null, ptr @zz_tmp, align 4
	br label %bb100.outer.outer

bb100.outer:		; preds = %bb2.i.i15.critedge, %bb100.outer.outer
	%state.0.ph = phi i32 [ 0, %bb100.outer.outer ], [ %iftmp.560.0, %bb2.i.i15.critedge ]		; <i32> [#uses=1]
	%cond = icmp eq i32 %state.0.ph, 1		; <i1> [#uses=1]
	br i1 %cond, label %bb2.i8, label %bb2.i84
}
