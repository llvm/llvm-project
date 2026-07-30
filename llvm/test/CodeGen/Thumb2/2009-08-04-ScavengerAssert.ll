; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -relocation-model=pic -frame-pointer=all -O3

	%0 = type { i16, i8, i8 }		; type %0
	%1 = type { [2 x i32], [2 x i32] }		; type %1
	%2 = type { %struct.GAP }		; type %2
	%3 = type { ptr }		; type %3
	%4 = type { i8, i8, i16, i8, i8, i8, i8 }		; type %4
	%5 = type { i8, i8, i8, i8 }		; type %5
	%struct.COMPOSITE = type { i8, i16, i16 }
	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.FILE_POS = type { i8, i8, i16, i32 }
	%struct.FIRST_UNION = type { %struct.FILE_POS }
	%struct.FONT_INFO = type { ptr, ptr, ptr, ptr, i32, ptr, ptr, i16, i16, ptr, ptr, ptr, ptr }
	%struct.FOURTH_UNION = type { %struct.STYLE }
	%struct.GAP = type { i8, i8, i16 }
	%struct.LIST = type { ptr, ptr }
	%struct.SECOND_UNION = type { %0 }
	%struct.STYLE = type { %2, %2, i16, i16, i32 }
	%struct.THIRD_UNION = type { %1 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
	%struct.head_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, ptr, %3, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32 }
	%struct.metrics = type { i16, i16, i16, i16, i16 }
	%struct.rec = type { %struct.head_type }
@.str24239 = external constant [20 x i8], align 1		; <ptr> [#uses=1]
@no_file_pos = external global %4		; <ptr> [#uses=1]
@.str19294 = external constant [9 x i8], align 1		; <ptr> [#uses=1]
@zz_lengths = external global [150 x i8]		; <ptr> [#uses=1]
@next_free.4772 = external global ptr		; <ptr> [#uses=3]
@top_free.4773 = external global ptr		; <ptr> [#uses=2]
@.str1575 = external constant [32 x i8], align 1		; <ptr> [#uses=1]
@zz_free = external global [524 x ptr]		; <ptr> [#uses=2]
@zz_hold = external global ptr		; <ptr> [#uses=5]
@zz_tmp = external global ptr		; <ptr> [#uses=2]
@zz_res = external global ptr		; <ptr> [#uses=2]
@xx_link = external global ptr		; <ptr> [#uses=2]
@font_count = external global i32		; <ptr> [#uses=1]
@.str81872 = external constant [10 x i8], align 1		; <ptr> [#uses=1]
@.str101874 = external constant [30 x i8], align 1		; <ptr> [#uses=1]
@.str111875 = external constant [17 x i8], align 1		; <ptr> [#uses=1]
@.str141878 = external constant [27 x i8], align 1		; <ptr> [#uses=1]
@out_fp = external global ptr		; <ptr> [#uses=3]
@.str192782 = external constant [17 x i8], align 1		; <ptr> [#uses=1]
@cpexists = external global i32		; <ptr> [#uses=2]
@.str212784 = external constant [17 x i8], align 1		; <ptr> [#uses=1]
@currentfont = external global i32		; <ptr> [#uses=3]
@wordcount = external global i32		; <ptr> [#uses=1]
@needs = external global ptr		; <ptr> [#uses=1]
@.str742838 = external constant [6 x i8], align 1		; <ptr> [#uses=1]
@.str752839 = external constant [10 x i8], align 1		; <ptr> [#uses=1]
@.str1802944 = external constant [40 x i8], align 1		; <ptr> [#uses=1]
@.str1822946 = external constant [8 x i8], align 1		; <ptr> [#uses=1]
@.str1842948 = external constant [11 x i8], align 1		; <ptr> [#uses=1]
@.str1852949 = external constant [23 x i8], align 1		; <ptr> [#uses=1]
@.str1872951 = external constant [17 x i8], align 1		; <ptr> [#uses=1]
@.str1932957 = external constant [26 x i8], align 1		; <ptr> [#uses=1]

declare i32 @fprintf(ptr nocapture, ptr nocapture, ...) nounwind

declare i32 @"\01_fwrite"(ptr, i32, i32, ptr)

declare i32 @remove(ptr nocapture) nounwind

declare ptr @OpenIncGraphicFile(ptr, i8 zeroext, ptr nocapture, ptr, ptr nocapture) nounwind

declare ptr @MakeWord(i32, ptr nocapture, ptr) nounwind

declare void @Error(i32, i32, ptr, i32, ptr, ...) nounwind

declare i32 @"\01_fputs"(ptr, ptr)

declare noalias ptr @calloc(i32, i32) nounwind

declare ptr @fgets(ptr, i32, ptr nocapture) nounwind

define void @PS_PrintGraphicInclude(ptr %x, i32 %colmark, i32 %rowmark) nounwind {
entry:
	%buff = alloca [512 x i8], align 4		; <ptr> [#uses=5]
	%0 = getelementptr %struct.rec, ptr %x, i32 0, i32 0, i32 1, i32 0, i32 0		; <ptr> [#uses=2]
	%1 = load i8, ptr %0, align 4		; <i8> [#uses=1]
	%2 = add i8 %1, -94		; <i8> [#uses=1]
	%3 = icmp ugt i8 %2, 1		; <i1> [#uses=1]
	br i1 %3, label %bb, label %bb1

bb:		; preds = %entry
	br label %bb1

bb1:		; preds = %bb, %entry
	%4 = getelementptr %struct.rec, ptr %x, i32 0, i32 0, i32 2		; <ptr> [#uses=1]
	%5 = getelementptr %4, ptr %4, i32 0, i32 1		; <ptr> [#uses=1]
	%6 = load i8, ptr %5, align 1		; <i8> [#uses=1]
	%7 = icmp eq i8 %6, 0		; <i1> [#uses=1]
	br i1 %7, label %bb2, label %bb3

bb2:		; preds = %bb1
	call  void (i32, i32, ptr, i32, ptr, ...) @Error(i32 1, i32 2, ptr @.str24239, i32 0, ptr @no_file_pos, ptr @.str1802944) nounwind
	br label %bb3

bb3:		; preds = %bb2, %bb1
	%8 = load ptr, ptr undef, align 4		; <ptr> [#uses=0]
	br label %bb5

bb5:		; preds = %bb5, %bb3
	%y.0 = load ptr, ptr null		; <ptr> [#uses=2]
	br i1 false, label %bb5, label %bb6

bb6:		; preds = %bb5
	%9 = load i8, ptr %0, align 4		; <i8> [#uses=1]
	%10 = getelementptr %struct.rec, ptr %y.0, i32 0, i32 0, i32 1, i32 0		; <ptr> [#uses=1]
	%11 = call  ptr @OpenIncGraphicFile(ptr undef, i8 zeroext %9, ptr null, ptr %10, ptr undef) nounwind		; <ptr> [#uses=4]
	br i1 false, label %bb7, label %bb8

bb7:		; preds = %bb6
	unreachable

bb8:		; preds = %bb6
	%12 = and i32 undef, 4095		; <i32> [#uses=2]
	%13 = load i32, ptr @currentfont, align 4		; <i32> [#uses=0]
	br i1 false, label %bb10, label %bb9

bb9:		; preds = %bb8
	%14 = icmp ult i32 0, %12		; <i1> [#uses=1]
	br i1 %14, label %bb.i, label %FontHalfXHeight.exit

bb.i:		; preds = %bb9
	call  void (i32, i32, ptr, i32, ptr, ...) @Error(i32 1, i32 2, ptr @.str24239, i32 0, ptr @no_file_pos, ptr @.str111875) nounwind
	%.pre186 = load i32, ptr @currentfont, align 4		; <i32> [#uses=1]
	br label %FontHalfXHeight.exit

FontHalfXHeight.exit:		; preds = %bb.i, %bb9
	%15 = phi i32 [ %.pre186, %bb.i ], [ %12, %bb9 ]		; <i32> [#uses=1]
	br i1 false, label %bb.i1, label %bb1.i

bb.i1:		; preds = %FontHalfXHeight.exit
	br label %bb1.i

bb1.i:		; preds = %bb.i1, %FontHalfXHeight.exit
	br i1 undef, label %bb2.i, label %FontSize.exit

bb2.i:		; preds = %bb1.i
	call  void (i32, i32, ptr, i32, ptr, ...) @Error(i32 37, i32 61, ptr @.str101874, i32 1, ptr null) nounwind
	unreachable

FontSize.exit:		; preds = %bb1.i
	%16 = getelementptr %struct.FONT_INFO, ptr undef, i32 %15, i32 5		; <ptr> [#uses=0]
	%17 = load i32, ptr undef, align 4		; <i32> [#uses=1]
	%18 = load i32, ptr @currentfont, align 4		; <i32> [#uses=2]
	%19 = load i32, ptr @font_count, align 4		; <i32> [#uses=1]
	%20 = icmp ult i32 %19, %18		; <i1> [#uses=1]
	br i1 %20, label %bb.i5, label %FontName.exit

bb.i5:		; preds = %FontSize.exit
	call  void (i32, i32, ptr, i32, ptr, ...) @Error(i32 1, i32 2, ptr @.str24239, i32 0, ptr @no_file_pos, ptr @.str81872) nounwind
	br label %FontName.exit

FontName.exit:		; preds = %bb.i5, %FontSize.exit
	%21 = phi ptr [ undef, %bb.i5 ], [ undef, %FontSize.exit ]		; <ptr> [#uses=1]
	%22 = getelementptr %struct.FONT_INFO, ptr %21, i32 %18, i32 5		; <ptr> [#uses=0]
	%23 = call  i32 (ptr, ptr, ...) @fprintf(ptr undef, ptr @.str1822946, i32 %17, ptr null) nounwind		; <i32> [#uses=0]
	br label %bb10

bb10:		; preds = %FontName.exit, %bb8
	%24 = call  i32 @"\01_fwrite"(ptr @.str1842948, i32 1, i32 10, ptr undef) nounwind		; <i32> [#uses=0]
	%25 = sub i32 %rowmark, undef		; <i32> [#uses=1]
	%26 = load ptr, ptr @out_fp, align 4		; <ptr> [#uses=1]
	%27 = call  i32 (ptr, ptr, ...) @fprintf(ptr %26, ptr @.str212784, i32 undef, i32 %25) nounwind		; <i32> [#uses=0]
	store i32 0, ptr @cpexists, align 4
	%28 = call  i32 (ptr, ptr, ...) @fprintf(ptr undef, ptr @.str192782, double 2.000000e+01, double 2.000000e+01) nounwind		; <i32> [#uses=0]
	%29 = getelementptr %struct.rec, ptr %y.0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	%30 = load i32, ptr %29, align 4		; <i32> [#uses=1]
	%31 = sub i32 0, %30		; <i32> [#uses=1]
	%32 = load i32, ptr undef, align 4		; <i32> [#uses=1]
	%33 = sub i32 0, %32		; <i32> [#uses=1]
	%34 = load ptr, ptr @out_fp, align 4		; <ptr> [#uses=1]
	%35 = call  i32 (ptr, ptr, ...) @fprintf(ptr %34, ptr @.str212784, i32 %31, i32 %33) nounwind		; <i32> [#uses=0]
	store i32 0, ptr @cpexists, align 4
	%36 = load ptr, ptr null, align 4		; <ptr> [#uses=1]
	%37 = getelementptr %struct.rec, ptr %36, i32 0, i32 0, i32 4		; <ptr> [#uses=1]
	%38 = call  i32 (ptr, ptr, ...) @fprintf(ptr undef, ptr @.str1852949, ptr %37) nounwind		; <i32> [#uses=0]
	%buff14 = getelementptr [512 x i8], ptr %buff, i32 0, i32 0		; <ptr> [#uses=5]
	%39 = call  ptr @fgets(ptr %buff14, i32 512, ptr %11) nounwind		; <ptr> [#uses=0]
	%iftmp.506.0 = select i1 undef, i32 2, i32 0		; <i32> [#uses=1]
	%40 = getelementptr [512 x i8], ptr %buff, i32 0, i32 26		; <ptr> [#uses=1]
	br label %bb100.outer.outer

bb100.outer.outer:		; preds = %bb83, %bb10
	%state.0.ph.ph = phi i32 [ %iftmp.506.0, %bb10 ], [ undef, %bb83 ]		; <i32> [#uses=1]
	%x_addr.0.ph.ph = phi ptr [ %x, %bb10 ], [ %70, %bb83 ]		; <ptr> [#uses=1]
	%41 = getelementptr %struct.rec, ptr %x_addr.0.ph.ph, i32 0, i32 0, i32 1, i32 0		; <ptr> [#uses=0]
	br label %bb100.outer

bb.i80:		; preds = %bb3.i85
	%42 = icmp eq i8 %43, %45		; <i1> [#uses=1]
	%indvar.next.i79 = add i32 %indvar.i81, 1		; <i32> [#uses=1]
	br i1 %42, label %bb2.i84, label %bb2.i51

bb2.i84:		; preds = %bb100.outer, %bb.i80
	%indvar.i81 = phi i32 [ %indvar.next.i79, %bb.i80 ], [ 0, %bb100.outer ]		; <i32> [#uses=3]
	%pp.0.i82 = getelementptr [27 x i8], ptr @.str141878, i32 0, i32 %indvar.i81		; <ptr> [#uses=2]
	%sp.0.i83 = getelementptr [512 x i8], ptr %buff, i32 0, i32 %indvar.i81		; <ptr> [#uses=1]
	%43 = load i8, ptr %sp.0.i83, align 1		; <i8> [#uses=2]
	%44 = icmp eq i8 %43, 0		; <i1> [#uses=1]
	br i1 %44, label %StringBeginsWith.exit88thread-split, label %bb3.i85

bb3.i85:		; preds = %bb2.i84
	%45 = load i8, ptr %pp.0.i82, align 1		; <i8> [#uses=3]
	%46 = icmp eq i8 %45, 0		; <i1> [#uses=1]
	br i1 %46, label %StringBeginsWith.exit88, label %bb.i80

StringBeginsWith.exit88thread-split:		; preds = %bb2.i84
	%.pr = load i8, ptr %pp.0.i82		; <i8> [#uses=1]
	br label %StringBeginsWith.exit88

StringBeginsWith.exit88:		; preds = %StringBeginsWith.exit88thread-split, %bb3.i85
	%47 = phi i8 [ %.pr, %StringBeginsWith.exit88thread-split ], [ %45, %bb3.i85 ]		; <i8> [#uses=1]
	%phitmp91 = icmp eq i8 %47, 0		; <i1> [#uses=1]
	br i1 %phitmp91, label %bb3.i77, label %bb2.i51

bb2.i.i68:		; preds = %bb3.i77
	br i1 false, label %bb2.i51, label %bb2.i75

bb2.i75:		; preds = %bb2.i.i68
	br label %bb3.i77

bb3.i77:		; preds = %bb2.i75, %StringBeginsWith.exit88
	%sp.0.i76 = getelementptr [512 x i8], ptr %buff, i32 0, i32 undef		; <ptr> [#uses=1]
	%48 = load i8, ptr %sp.0.i76, align 1		; <i8> [#uses=1]
	%49 = icmp eq i8 %48, 0		; <i1> [#uses=1]
	br i1 %49, label %bb24, label %bb2.i.i68

bb24:		; preds = %bb3.i77
	%50 = call  ptr @MakeWord(i32 11, ptr %40, ptr @no_file_pos) nounwind		; <ptr> [#uses=0]
	%51 = load i8, ptr @zz_lengths, align 4		; <i8> [#uses=1]
	%52 = zext i8 %51 to i32		; <i32> [#uses=2]
	%53 = getelementptr [524 x ptr], ptr @zz_free, i32 0, i32 %52		; <ptr> [#uses=2]
	%54 = load ptr, ptr %53, align 4		; <ptr> [#uses=3]
	%55 = icmp eq ptr %54, null		; <i1> [#uses=1]
	br i1 %55, label %bb27, label %bb28

bb27:		; preds = %bb24
	br i1 undef, label %bb.i56, label %GetMemory.exit62

bb.i56:		; preds = %bb27
	br i1 undef, label %bb1.i58, label %bb2.i60

bb1.i58:		; preds = %bb.i56
	call  void (i32, i32, ptr, i32, ptr, ...) @Error(i32 31, i32 1, ptr @.str1575, i32 1, ptr @no_file_pos) nounwind
	br label %bb2.i60

bb2.i60:		; preds = %bb1.i58, %bb.i56
	%.pre1.i59 = phi ptr [ undef, %bb1.i58 ], [ undef, %bb.i56 ]		; <ptr> [#uses=1]
	store ptr undef, ptr @top_free.4773, align 4
	br label %GetMemory.exit62

GetMemory.exit62:		; preds = %bb2.i60, %bb27
	%56 = phi ptr [ %.pre1.i59, %bb2.i60 ], [ undef, %bb27 ]		; <ptr> [#uses=1]
	%57 = getelementptr ptr, ptr %56, i32 %52		; <ptr> [#uses=1]
	store ptr %57, ptr @next_free.4772, align 4
	store ptr undef, ptr @zz_hold, align 4
	br label %bb29

bb28:		; preds = %bb24
	store ptr %54, ptr @zz_hold, align 4
	%58 = load ptr, ptr null, align 4		; <ptr> [#uses=1]
	store ptr %58, ptr %53, align 4
	br label %bb29

bb29:		; preds = %bb28, %GetMemory.exit62
	%.pre184 = phi ptr [ %54, %bb28 ], [ undef, %GetMemory.exit62 ]		; <ptr> [#uses=3]
	store i8 0, ptr undef
	store ptr %.pre184, ptr @xx_link, align 4
	br i1 undef, label %bb35, label %bb31

bb31:		; preds = %bb29
	store ptr %.pre184, ptr undef
	br label %bb35

bb35:		; preds = %bb31, %bb29
	br i1 undef, label %bb41, label %bb37

bb37:		; preds = %bb35
	%59 = load ptr, ptr null, align 4		; <ptr> [#uses=1]
	store ptr %59, ptr undef
	store ptr undef, ptr null
	store ptr %.pre184, ptr null, align 4
	br label %bb41

bb41:		; preds = %bb37, %bb35
	%60 = call  ptr @fgets(ptr %buff14, i32 512, ptr %11) nounwind		; <ptr> [#uses=1]
	%61 = icmp eq ptr %60, null		; <i1> [#uses=1]
	%iftmp.554.0 = select i1 %61, i32 2, i32 1		; <i32> [#uses=1]
	br label %bb100.outer

bb.i47:		; preds = %bb3.i52
	%62 = icmp eq i8 %63, %64		; <i1> [#uses=1]
	br i1 %62, label %bb2.i51, label %bb2.i41

bb2.i51:		; preds = %bb.i47, %bb2.i.i68, %StringBeginsWith.exit88, %bb.i80
	%pp.0.i49 = getelementptr [17 x i8], ptr @.str1872951, i32 0, i32 0		; <ptr> [#uses=1]
	%63 = load i8, ptr null, align 1		; <i8> [#uses=1]
	br i1 false, label %StringBeginsWith.exit55thread-split, label %bb3.i52

bb3.i52:		; preds = %bb2.i51
	%64 = load i8, ptr %pp.0.i49, align 1		; <i8> [#uses=1]
	br i1 false, label %StringBeginsWith.exit55, label %bb.i47

StringBeginsWith.exit55thread-split:		; preds = %bb2.i51
	br label %StringBeginsWith.exit55

StringBeginsWith.exit55:		; preds = %StringBeginsWith.exit55thread-split, %bb3.i52
	br i1 false, label %bb49, label %bb2.i41

bb49:		; preds = %StringBeginsWith.exit55
	br label %bb2.i41

bb2.i41:		; preds = %bb2.i41, %bb49, %StringBeginsWith.exit55, %bb.i47
	br i1 false, label %bb2.i41, label %bb2.i.i15

bb2.i.i15:		; preds = %bb2.i41
	%pp.0.i.i13 = getelementptr [6 x i8], ptr @.str742838, i32 0, i32 0		; <ptr> [#uses=1]
	br i1 false, label %StringBeginsWith.exitthread-split.i18, label %bb3.i.i16

bb3.i.i16:		; preds = %bb2.i.i15
	%65 = load i8, ptr %pp.0.i.i13, align 1		; <i8> [#uses=1]
	br label %StringBeginsWith.exit.i20

StringBeginsWith.exitthread-split.i18:		; preds = %bb2.i.i15
	br label %StringBeginsWith.exit.i20

StringBeginsWith.exit.i20:		; preds = %StringBeginsWith.exitthread-split.i18, %bb3.i.i16
	%66 = phi i8 [ undef, %StringBeginsWith.exitthread-split.i18 ], [ %65, %bb3.i.i16 ]		; <i8> [#uses=1]
	%phitmp.i19 = icmp eq i8 %66, 0		; <i1> [#uses=1]
	br i1 %phitmp.i19, label %bb58, label %bb2.i6.i26

bb2.i6.i26:		; preds = %bb2.i6.i26, %StringBeginsWith.exit.i20
	%indvar.i3.i23 = phi i32 [ %indvar.next.i1.i21, %bb2.i6.i26 ], [ 0, %StringBeginsWith.exit.i20 ]		; <i32> [#uses=3]
	%sp.0.i5.i25 = getelementptr [512 x i8], ptr %buff, i32 0, i32 %indvar.i3.i23		; <ptr> [#uses=0]
	%pp.0.i4.i24 = getelementptr [10 x i8], ptr @.str752839, i32 0, i32 %indvar.i3.i23		; <ptr> [#uses=1]
	%67 = load i8, ptr %pp.0.i4.i24, align 1		; <i8> [#uses=0]
	%indvar.next.i1.i21 = add i32 %indvar.i3.i23, 1		; <i32> [#uses=1]
	br i1 undef, label %bb2.i6.i26, label %bb55

bb55:		; preds = %bb2.i6.i26
	%68 = call  i32 @"\01_fputs"(ptr %buff14, ptr undef) nounwind		; <i32> [#uses=0]
	unreachable

bb58:		; preds = %StringBeginsWith.exit.i20
	%69 = call  ptr @fgets(ptr %buff14, i32 512, ptr %11) nounwind		; <ptr> [#uses=0]
	%iftmp.560.0 = select i1 undef, i32 2, i32 0		; <i32> [#uses=1]
	br label %bb100.outer

bb.i7:		; preds = %bb3.i
	br i1 false, label %bb2.i8, label %bb2.i.i

bb2.i8:		; preds = %bb100.outer, %bb.i7
	br i1 undef, label %StringBeginsWith.exitthread-split, label %bb3.i

bb3.i:		; preds = %bb2.i8
	br i1 undef, label %StringBeginsWith.exit, label %bb.i7

StringBeginsWith.exitthread-split:		; preds = %bb2.i8
	br label %StringBeginsWith.exit

StringBeginsWith.exit:		; preds = %StringBeginsWith.exitthread-split, %bb3.i
	%phitmp93 = icmp eq i8 undef, 0		; <i1> [#uses=1]
	br i1 %phitmp93, label %bb66, label %bb2.i.i

bb66:		; preds = %StringBeginsWith.exit
	%70 = call  ptr @MakeWord(i32 11, ptr undef, ptr @no_file_pos) nounwind		; <ptr> [#uses=4]
	%71 = load i8, ptr @zz_lengths, align 4		; <i8> [#uses=1]
	%72 = zext i8 %71 to i32		; <i32> [#uses=2]
	%73 = getelementptr [524 x ptr], ptr @zz_free, i32 0, i32 %72		; <ptr> [#uses=2]
	%74 = load ptr, ptr %73, align 4		; <ptr> [#uses=3]
	%75 = icmp eq ptr %74, null		; <i1> [#uses=1]
	br i1 %75, label %bb69, label %bb70

bb69:		; preds = %bb66
	br i1 undef, label %bb.i2, label %GetMemory.exit

bb.i2:		; preds = %bb69
	%76 = call  noalias ptr @calloc(i32 1020, i32 4) nounwind		; <ptr> [#uses=1]
	store ptr %76, ptr @next_free.4772, align 4
	br i1 undef, label %bb1.i3, label %bb2.i4

bb1.i3:		; preds = %bb.i2
	call  void (i32, i32, ptr, i32, ptr, ...) @Error(i32 31, i32 1, ptr @.str1575, i32 1, ptr @no_file_pos) nounwind
	br label %bb2.i4

bb2.i4:		; preds = %bb1.i3, %bb.i2
	%.pre1.i = phi ptr [ undef, %bb1.i3 ], [ %76, %bb.i2 ]		; <ptr> [#uses=1]
	%77 = phi ptr [ undef, %bb1.i3 ], [ %76, %bb.i2 ]		; <ptr> [#uses=1]
	%78 = getelementptr ptr, ptr %77, i32 1020		; <ptr> [#uses=1]
	store ptr %78, ptr @top_free.4773, align 4
	br label %GetMemory.exit

GetMemory.exit:		; preds = %bb2.i4, %bb69
	%79 = phi ptr [ %.pre1.i, %bb2.i4 ], [ undef, %bb69 ]		; <ptr> [#uses=2]
	%80 = getelementptr ptr, ptr %79, i32 %72		; <ptr> [#uses=1]
	store ptr %80, ptr @next_free.4772, align 4
	store ptr %79, ptr @zz_hold, align 4
	br label %bb71

bb70:		; preds = %bb66
	%81 = load ptr, ptr null, align 4		; <ptr> [#uses=1]
	store ptr %81, ptr %73, align 4
	br label %bb71

bb71:		; preds = %bb70, %GetMemory.exit
	%.pre185 = phi ptr [ %74, %bb70 ], [ %79, %GetMemory.exit ]		; <ptr> [#uses=8]
	%82 = phi ptr [ %74, %bb70 ], [ %79, %GetMemory.exit ]		; <ptr> [#uses=1]
	%83 = getelementptr %struct.rec, ptr %82, i32 0, i32 0, i32 1, i32 0, i32 0		; <ptr> [#uses=0]
	%84 = getelementptr %struct.rec, ptr %.pre185, i32 0, i32 0, i32 0, i32 1, i32 1		; <ptr> [#uses=0]
	%85 = getelementptr %struct.rec, ptr %.pre185, i32 0, i32 0, i32 0, i32 1, i32 0		; <ptr> [#uses=1]
	store ptr %.pre185, ptr @xx_link, align 4
	store ptr %.pre185, ptr @zz_res, align 4
	%86 = load ptr, ptr @needs, align 4		; <ptr> [#uses=2]
	store ptr %86, ptr @zz_hold, align 4
	br i1 false, label %bb77, label %bb73

bb73:		; preds = %bb71
	%87 = getelementptr %struct.rec, ptr %86, i32 0, i32 0, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	store ptr null, ptr @zz_tmp, align 4
	store ptr %.pre185, ptr %87
	store ptr %.pre185, ptr undef, align 4
	br label %bb77

bb77:		; preds = %bb73, %bb71
	store ptr %.pre185, ptr @zz_res, align 4
	store ptr %70, ptr @zz_hold, align 4
	br i1 undef, label %bb83, label %bb79

bb79:		; preds = %bb77
	%88 = getelementptr %struct.rec, ptr %70, i32 0, i32 0, i32 0, i32 1, i32 0		; <ptr> [#uses=1]
	store ptr null, ptr @zz_tmp, align 4
	%89 = load ptr, ptr %85, align 4		; <ptr> [#uses=1]
	store ptr %89, ptr %88
	%90 = getelementptr %struct.rec, ptr undef, i32 0, i32 0, i32 0, i32 1, i32 1		; <ptr> [#uses=1]
	store ptr %70, ptr %90, align 4
	store ptr %.pre185, ptr undef, align 4
	br label %bb83

bb83:		; preds = %bb79, %bb77
	br label %bb100.outer.outer

bb.i.i:		; preds = %bb3.i.i
	br i1 undef, label %bb2.i.i, label %bb2.i6.i

bb2.i.i:		; preds = %bb.i.i, %StringBeginsWith.exit, %bb.i7
	br i1 undef, label %StringBeginsWith.exitthread-split.i, label %bb3.i.i

bb3.i.i:		; preds = %bb2.i.i
	br i1 undef, label %StringBeginsWith.exit.i, label %bb.i.i

StringBeginsWith.exitthread-split.i:		; preds = %bb2.i.i
	br label %StringBeginsWith.exit.i

StringBeginsWith.exit.i:		; preds = %StringBeginsWith.exitthread-split.i, %bb3.i.i
	br i1 false, label %bb94, label %bb2.i6.i

bb.i2.i:		; preds = %bb3.i7.i
	br i1 false, label %bb2.i6.i, label %bb91

bb2.i6.i:		; preds = %bb.i2.i, %StringBeginsWith.exit.i, %bb.i.i
	br i1 undef, label %strip_out.exitthread-split, label %bb3.i7.i

bb3.i7.i:		; preds = %bb2.i6.i
	%91 = load i8, ptr undef, align 1		; <i8> [#uses=1]
	br i1 undef, label %strip_out.exit, label %bb.i2.i

strip_out.exitthread-split:		; preds = %bb2.i6.i
	%.pr100 = load i8, ptr undef		; <i8> [#uses=1]
	br label %strip_out.exit

strip_out.exit:		; preds = %strip_out.exitthread-split, %bb3.i7.i
	%92 = phi i8 [ %.pr100, %strip_out.exitthread-split ], [ %91, %bb3.i7.i ]		; <i8> [#uses=0]
	br i1 undef, label %bb94, label %bb91

bb91:		; preds = %strip_out.exit, %bb.i2.i
	unreachable

bb94:		; preds = %strip_out.exit, %StringBeginsWith.exit.i
	%93 = call  ptr @fgets(ptr %buff14, i32 512, ptr %11) nounwind		; <ptr> [#uses=0]
	unreachable

bb100.outer:		; preds = %bb58, %bb41, %bb100.outer.outer
	%state.0.ph = phi i32 [ %state.0.ph.ph, %bb100.outer.outer ], [ %iftmp.560.0, %bb58 ], [ %iftmp.554.0, %bb41 ]		; <i32> [#uses=1]
	switch i32 %state.0.ph, label %bb2.i84 [
		i32 2, label %bb101.split
		i32 1, label %bb2.i8
	]

bb101.split:		; preds = %bb100.outer
	%94 = icmp eq i32 undef, 0		; <i1> [#uses=1]
	br i1 %94, label %bb103, label %bb102

bb102:		; preds = %bb101.split
	%95 = call  i32 @remove(ptr @.str19294) nounwind		; <i32> [#uses=0]
	unreachable

bb103:		; preds = %bb101.split
	%96 = load ptr, ptr @out_fp, align 4		; <ptr> [#uses=1]
	%97 = call  i32 (ptr, ptr, ...) @fprintf(ptr %96, ptr @.str1932957) nounwind		; <i32> [#uses=0]
	store i32 0, ptr @wordcount, align 4
	ret void
}
