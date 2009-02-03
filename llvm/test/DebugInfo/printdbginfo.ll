; RUN: llvm-as < %s | opt -print-dbginfo -disable-output > %t1
; RUN: %prcontext {function name: Bar::bar return type: int at line 12} 1 < %t1 | grep {(tst.cpp:14)}
; RUN: %prcontext {%%tmp1} 1 < %t1 | grep -E {variable tmp.+at line 23}
; RUN: %prcontext {; tst.cpp:24} 2 < %t1 | grep {%%6}
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8* }
	%llvm.dbg.compositetype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
	%struct.Bar = type { %struct.Foo, i32 }
	%struct.Foo = type { i32 }
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 393233, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([8 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [8 x i8] c"tst.cpp\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1 = internal constant [13 x i8] c"/home/edwin/\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5623) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@.str3 = internal constant [4 x i8] c"bar\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str4 = internal constant [9 x i8] c"Bar::bar\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str5 = internal constant [14 x i8] c"_ZN3Bar3barEv\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 393252, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str6, i32 0, i32 0), { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str6 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 393473, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([5 x i8]* @.str7, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=0]
@.str7 = internal constant [5 x i8] c"this\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 393231, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 64, i64 64, i64 0, i32 0, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.compositetype = internal constant %llvm.dbg.compositetype.type { i32 393235, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 5, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array36 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@.str8 = internal constant [4 x i8] c"Bar\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.derivedtype9 = internal constant %llvm.dbg.derivedtype.type { i32 393244, { }* null, i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype10 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.compositetype10 = internal constant %llvm.dbg.compositetype.type { i32 393235, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str11, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 1, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array22 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@.str11 = internal constant [4 x i8] c"Foo\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.derivedtype12 = internal constant %llvm.dbg.derivedtype.type { i32 393229, { }* null, i8* getelementptr ([7 x i8]* @.str13, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str13 = internal constant [7 x i8] c"FooVar\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram14 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str11, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str15, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype16 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str15 = internal constant [9 x i8] c"Foo::Foo\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compositetype16 = internal constant %llvm.dbg.compositetype.type { i32 393237, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 8, i64 8, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.derivedtype17 = internal constant %llvm.dbg.derivedtype.type { i32 393231, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 64, i64 64, i64 0, i32 0, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype10 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype18 = internal constant %llvm.dbg.derivedtype.type { i32 393232, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 64, i64 64, i64 0, i32 0, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype10 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype17 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype18 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.subprogram19 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str11, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str15, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype20 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.compositetype20 = internal constant %llvm.dbg.compositetype.type { i32 393237, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 8, i64 8, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array21 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.array21 = internal constant [2 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype17 to { }*) ], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.array22 = internal constant [3 x { }*] [ { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram14 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram19 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.derivedtype23 = internal constant %llvm.dbg.derivedtype.type { i32 393229, { }* null, i8* getelementptr ([7 x i8]* @.str24, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 6, i64 32, i64 32, i64 32, i32 1, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str24 = internal constant [7 x i8] c"BarVar\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram25 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str26, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype27 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str26 = internal constant [9 x i8] c"Bar::Bar\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compositetype27 = internal constant %llvm.dbg.compositetype.type { i32 393237, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 8, i64 8, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array29 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.derivedtype28 = internal constant %llvm.dbg.derivedtype.type { i32 393232, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 64, i64 64, i64 0, i32 0, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array29 = internal constant [3 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype28 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.subprogram30 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str26, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype31 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.compositetype31 = internal constant %llvm.dbg.compositetype.type { i32 393237, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 8, i64 8, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array32 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.array32 = internal constant [2 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) ], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.subprogram33 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype34 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.compositetype34 = internal constant %llvm.dbg.compositetype.type { i32 393237, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 8, i64 8, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array35 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.array35 = internal constant [2 x { }*] [ { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) ], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.array36 = internal constant [5 x { }*] [ { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype23 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram25 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram30 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram33 to { }*) ], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.variable37 = internal constant %llvm.dbg.variable.type { i32 393472, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([4 x i8]* @.str38, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 15, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=0]
@.str38 = internal constant [4 x i8] c"tmp\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.subprogram39 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str40, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str40, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str41, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 21, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str40 = internal constant [7 x i8] c"foobar\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str41 = internal constant [11 x i8] c"_Z6foobarv\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.variable42 = internal constant %llvm.dbg.variable.type { i32 393472, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram39 to { }*), i8* getelementptr ([4 x i8]* @.str38, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 23, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=0]

define i32 @_ZN3Bar3barEv(%struct.Bar* %this1) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	tail call void @llvm.dbg.stoppoint(i32 14, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.Bar* %this1, i64 0, i32 0, i32 0		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = icmp sgt i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb, label %bb3

bb:		; preds = %entry
	%3 = getelementptr %struct.Bar* %this1, i64 0, i32 1		; <i32*> [#uses=1]
	%4 = load i32* %3, align 4		; <i32> [#uses=1]
	%5 = shl i32 %4, 1		; <i32> [#uses=1]
	tail call void @llvm.dbg.stoppoint(i32 16, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb4

bb3:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 18, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb4

bb4:		; preds = %bb3, %bb
	%.0 = phi i32 [ 0, %bb3 ], [ %5, %bb ]		; <i32> [#uses=1]
	tail call void @llvm.dbg.stoppoint(i32 18, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret i32 %.0
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind

define %struct.Bar* @_Z6foobarv() {
entry:
	%retval = alloca %struct.Bar*		; <%struct.Bar**> [#uses=2]
	%tmp = alloca %struct.Bar*		; <%struct.Bar**> [#uses=3]
	%0 = alloca %struct.Bar*		; <%struct.Bar**> [#uses=2]
	%1 = alloca %struct.Bar*		; <%struct.Bar**> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram39 to { }*))
	%tmp1 = bitcast %struct.Bar** %tmp to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %tmp1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable42 to { }*))
	call void @llvm.dbg.stoppoint(i32 23, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = call i8* @_Znwm(i64 8)		; <i8*> [#uses=1]
	%3 = bitcast i8* %2 to %struct.Bar*		; <%struct.Bar*> [#uses=1]
	store %struct.Bar* %3, %struct.Bar** %1, align 8
	%4 = load %struct.Bar** %1, align 8		; <%struct.Bar*> [#uses=1]
	call void @_ZN3BarC1Ev(%struct.Bar* %4) nounwind
	%5 = load %struct.Bar** %1, align 8		; <%struct.Bar*> [#uses=1]
	store %struct.Bar* %5, %struct.Bar** %tmp, align 8
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = load %struct.Bar** %tmp, align 8		; <%struct.Bar*> [#uses=1]
	store %struct.Bar* %6, %struct.Bar** %0, align 8
	%7 = load %struct.Bar** %0, align 8		; <%struct.Bar*> [#uses=1]
	store %struct.Bar* %7, %struct.Bar** %retval, align 8
	br label %return

return:		; preds = %entry
	%retval2 = load %struct.Bar** %retval		; <%struct.Bar*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 24, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram39 to { }*))
	ret %struct.Bar* %retval2
}

declare i8* @_Znwm(i64)

declare void @_ZN3BarC1Ev(%struct.Bar*) nounwind
