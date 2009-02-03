; RUN: llvm-as < %s | llc -f -o /dev/null
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32, i8*, i8* }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, i8*, i8* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, i8*, i8* }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }*, i8*, i8* }
	%struct._RuneCharClass = type { [14 x i8], i32 }
	%struct._RuneEntry = type { i32, i32, i32, i32* }
	%struct._RuneLocale = type { [8 x i8], [32 x i8], i32 (i8*, i32, i8**)*, i32 (i32, i8*, i32, i8**)*, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, i8*, i32, i32, %struct._RuneCharClass* }
	%struct._RuneRange = type { i32, %struct._RuneEntry* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [4 x i8] c"x.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [5 x i8] c"/tmp\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str2 = internal constant [57 x i8] c"4.2.1 (Based on Apple Inc. build 5628) (LLVM build 9999)\00", section "llvm.metadata"		; <[57 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([57 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5, i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str4 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 21, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 false, i1 true, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str5 = internal constant [2 x i8] c"i\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([2 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 22, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str6 = internal constant [8 x i8] c"islower\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.subprogram9 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str6, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str6, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 267, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 true, i1 true, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str10 = internal constant [3 x i8] c"_c\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.variable11 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram9 to { }*), i8* getelementptr ([3 x i8]* @.str10, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 266, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str12 = internal constant [9 x i8] c"__istype\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.subprogram13 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str12, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str12, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 171, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 true, i1 true, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str14 = internal constant [19 x i8] c"__darwin_ct_rune_t\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@.str15 = internal constant [9 x i8] c"_types.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str16 = internal constant [18 x i8] c"/usr/include/i386\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str14, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 70, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i8* getelementptr ([9 x i8]* @.str15, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str16, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.variable17 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram13 to { }*), i8* getelementptr ([3 x i8]* @.str10, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 170, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str18 = internal constant [18 x i8] c"long unsigned int\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.basictype19 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str18, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7, i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str20 = internal constant [3 x i8] c"_f\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.variable21 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram13 to { }*), i8* getelementptr ([3 x i8]* @.str20, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 170, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype19 to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@_DefaultRuneLocale = external global %struct._RuneLocale		; <%struct._RuneLocale*> [#uses=1]
@.str22 = internal constant [8 x i8] c"isascii\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.subprogram23 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str22, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str22, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 153, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 true, i1 true, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable24 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram23 to { }*), i8* getelementptr ([3 x i8]* @.str10, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 152, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str25 = internal constant [8 x i8] c"toupper\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.subprogram26 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str25, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 316, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 true, i1 true, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable27 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram26 to { }*), i8* getelementptr ([3 x i8]* @.str10, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 315, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]

define i32 @main() nounwind {
entry:
	%retval = alloca i32		; <i32*> [#uses=1]
	%i = alloca i32		; <i32*> [#uses=16]
	%iftmp.5 = alloca i32		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%0 = bitcast i32* %i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	call void @llvm.dbg.stoppoint(i32 23, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 0, i32* %i, align 4
	br label %bb13
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 23, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb

bb:		; preds = %bb13, %1
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = load i32* %i, align 4		; <i32> [#uses=1]
	%3 = call i32 @islower(i32 %2) nounwind		; <i32> [#uses=1]
	%4 = icmp eq i32 %3, 0		; <i1> [#uses=1]
	br i1 %4, label %bb3, label %bb1
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb1

bb1:		; preds = %5, %bb
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = load i32* %i, align 4		; <i32> [#uses=1]
	%7 = icmp sle i32 %6, 96		; <i1> [#uses=1]
	br i1 %7, label %bb11, label %bb2
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb2

bb2:		; preds = %8, %bb1
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%9 = load i32* %i, align 4		; <i32> [#uses=1]
	%10 = icmp sgt i32 %9, 122		; <i1> [#uses=1]
	br i1 %10, label %bb11, label %bb3
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb3

bb3:		; preds = %11, %bb2, %bb
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%12 = load i32* %i, align 4		; <i32> [#uses=1]
	%13 = call i32 @islower(i32 %12) nounwind		; <i32> [#uses=1]
	%14 = icmp ne i32 %13, 0		; <i1> [#uses=1]
	br i1 %14, label %bb6, label %bb4
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb4

bb4:		; preds = %15, %bb3
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%16 = load i32* %i, align 4		; <i32> [#uses=1]
	%17 = icmp sle i32 %16, 96		; <i1> [#uses=1]
	br i1 %17, label %bb6, label %bb5
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb5

bb5:		; preds = %18, %bb4
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%19 = load i32* %i, align 4		; <i32> [#uses=1]
	%20 = icmp sle i32 %19, 122		; <i1> [#uses=1]
	br i1 %20, label %bb11, label %bb6
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb6

bb6:		; preds = %21, %bb5, %bb4, %bb3
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%22 = load i32* %i, align 4		; <i32> [#uses=1]
	%23 = call i32 @toupper(i32 %22) nounwind		; <i32> [#uses=1]
	%24 = load i32* %i, align 4		; <i32> [#uses=1]
	%25 = icmp sle i32 %24, 96		; <i1> [#uses=1]
	br i1 %25, label %bb9, label %bb7
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb7

bb7:		; preds = %26, %bb6
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%27 = load i32* %i, align 4		; <i32> [#uses=1]
	%28 = icmp sgt i32 %27, 122		; <i1> [#uses=1]
	br i1 %28, label %bb9, label %bb8
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8

bb8:		; preds = %29, %bb7
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%30 = load i32* %i, align 4		; <i32> [#uses=1]
	%31 = sub i32 %30, 32		; <i32> [#uses=1]
	store i32 %31, i32* %iftmp.5, align 4
	br label %bb10
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb9

bb9:		; preds = %32, %bb7, %bb6
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%33 = load i32* %i, align 4		; <i32> [#uses=1]
	store i32 %33, i32* %iftmp.5, align 4
	br label %bb10

bb10:		; preds = %bb9, %bb8
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%34 = load i32* %iftmp.5, align 4		; <i32> [#uses=1]
	%35 = icmp ne i32 %23, %34		; <i1> [#uses=1]
	br i1 %35, label %bb11, label %bb12
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb11

bb11:		; preds = %36, %bb10, %bb5, %bb2, %bb1
	call void @llvm.dbg.stoppoint(i32 26, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @exit(i32 2) noreturn nounwind
	unreachable
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 26, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb12

bb12:		; preds = %37, %bb10
	call void @llvm.dbg.stoppoint(i32 23, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%38 = load i32* %i, align 4		; <i32> [#uses=1]
	%39 = add i32 %38, 1		; <i32> [#uses=1]
	store i32 %39, i32* %i, align 4
	br label %bb13

bb13:		; preds = %bb12, %entry
	call void @llvm.dbg.stoppoint(i32 23, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%40 = load i32* %i, align 4		; <i32> [#uses=1]
	%41 = icmp sle i32 %40, 255		; <i1> [#uses=1]
	br i1 %41, label %bb, label %bb14
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 23, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb14

bb14:		; preds = %42, %bb13
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @exit(i32 0) noreturn nounwind
	unreachable

return:		; No predecessors!
	%retval15 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret i32 %retval15
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

define internal i32 @islower(i32 %_c) nounwind {
entry:
	%_c_addr = alloca i32		; <i32*> [#uses=3]
	%retval = alloca i32		; <i32*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram9 to { }*))
	%1 = bitcast i32* %_c_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable11 to { }*))
	store i32 %_c, i32* %_c_addr
	call void @llvm.dbg.stoppoint(i32 268, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = load i32* %_c_addr, align 4		; <i32> [#uses=1]
	%3 = call i32 @__istype(i32 %2, i32 4096) nounwind		; <i32> [#uses=1]
	store i32 %3, i32* %0, align 4
	%4 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %4, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 268, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram9 to { }*))
	ret i32 %retval1
}

define internal i32 @toupper(i32 %_c) nounwind {
entry:
	%_c_addr = alloca i32		; <i32*> [#uses=3]
	%retval = alloca i32		; <i32*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram26 to { }*))
	%1 = bitcast i32* %_c_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable27 to { }*))
	store i32 %_c, i32* %_c_addr
	call void @llvm.dbg.stoppoint(i32 317, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = load i32* %_c_addr, align 4		; <i32> [#uses=1]
	%3 = call i32 @__toupper(i32 %2) nounwind		; <i32> [#uses=1]
	store i32 %3, i32* %0, align 4
	%4 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %4, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 317, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram26 to { }*))
	ret i32 %retval1
}

declare void @exit(i32) noreturn nounwind

declare void @llvm.dbg.region.end({ }*) nounwind

define internal i32 @__istype(i32 %_c, i32 %_f) nounwind {
entry:
	%_c_addr = alloca i32		; <i32*> [#uses=5]
	%_f_addr = alloca i32		; <i32*> [#uses=4]
	%retval = alloca i32		; <i32*> [#uses=2]
	%iftmp.0 = alloca i32		; <i32*> [#uses=3]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram13 to { }*))
	%1 = bitcast i32* %_c_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable17 to { }*))
	store i32 %_c, i32* %_c_addr
	%2 = bitcast i32* %_f_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable21 to { }*))
	store i32 %_f, i32* %_f_addr
	call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = load i32* %_c_addr, align 4		; <i32> [#uses=1]
	%4 = call i32 @isascii(i32 %3) nounwind		; <i32> [#uses=1]
	%5 = icmp ne i32 %4, 0		; <i1> [#uses=1]
	br i1 %5, label %bb, label %bb1
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb

bb:		; preds = %6, %entry
	call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%7 = load i32* %_c_addr, align 4		; <i32> [#uses=1]
	%8 = getelementptr [256 x i32]* getelementptr (%struct._RuneLocale* @_DefaultRuneLocale, i32 0, i32 5), i32 0, i32 %7		; <i32*> [#uses=1]
	%9 = load i32* %8, align 4		; <i32> [#uses=1]
	%10 = load i32* %_f_addr, align 4		; <i32> [#uses=1]
	%11 = and i32 %9, %10		; <i32> [#uses=1]
	%12 = icmp ne i32 %11, 0		; <i1> [#uses=1]
	%13 = zext i1 %12 to i32		; <i32> [#uses=1]
	store i32 %13, i32* %iftmp.0, align 4
	br label %bb2
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb1

bb1:		; preds = %14, %entry
	call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%15 = load i32* %_c_addr, align 4		; <i32> [#uses=1]
	%16 = load i32* %_f_addr, align 4		; <i32> [#uses=1]
	%17 = call i32 @__maskrune(i32 %15, i32 %16) nounwind		; <i32> [#uses=1]
	%18 = icmp ne i32 %17, 0		; <i1> [#uses=1]
	%19 = zext i1 %18 to i32		; <i32> [#uses=1]
	store i32 %19, i32* %iftmp.0, align 4
	br label %bb2

bb2:		; preds = %bb1, %bb
	call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%20 = load i32* %iftmp.0, align 4		; <i32> [#uses=1]
	store i32 %20, i32* %0, align 4
	%21 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %21, i32* %retval, align 4
	br label %return

return:		; preds = %bb2
	%retval3 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram13 to { }*))
	ret i32 %retval3
}

define internal i32 @isascii(i32 %_c) nounwind {
entry:
	%_c_addr = alloca i32		; <i32*> [#uses=3]
	%retval = alloca i32		; <i32*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram23 to { }*))
	%1 = bitcast i32* %_c_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable24 to { }*))
	store i32 %_c, i32* %_c_addr
	call void @llvm.dbg.stoppoint(i32 154, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = load i32* %_c_addr, align 4		; <i32> [#uses=1]
	%3 = and i32 %2, -128		; <i32> [#uses=1]
	%4 = icmp eq i32 %3, 0		; <i1> [#uses=1]
	%5 = zext i1 %4 to i32		; <i32> [#uses=1]
	store i32 %5, i32* %0, align 4
	%6 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %6, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 154, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram23 to { }*))
	ret i32 %retval1
}

declare i32 @__maskrune(i32, i32)

declare i32 @__toupper(i32)
