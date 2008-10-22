; RUN: llvm-as < %s | llc
; PR2885

;; NOTE: This generates bad debug info in this case! But that's better than
;; ICEing.

; ModuleID = 'bug.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str3, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 14, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 true, i1 true }		; <%llvm.dbg.subprogram.type*> [#uses=0]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 46 }		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 393233, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([7 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0) }		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [7 x i8] c"die.cc\00"		; <[7 x i8]*> [#uses=1]
@.str1 = internal constant [16 x i8] c"/home/nicholas/\00"		; <[16 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5623) (LLVM build)\00"		; <[52 x i8]*> [#uses=1]
@.str3 = internal constant [11 x i8] c"AssertFail\00"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 393252, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str4 = internal constant [4 x i8] c"int\00"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.subprogram5 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str6, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str6, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 19, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype7 to { }*), i1 true, i1 true }		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str6 = internal constant [7 x i8] c"FooOne\00"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.basictype7 = internal constant %llvm.dbg.basictype.type { i32 393252, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str8, i32 0, i32 0), { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str8 = internal constant [9 x i8] c"long int\00"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 393473, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram5 to { }*), i8* getelementptr ([6 x i8]* @.str9, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 19, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype10 to { }*) }		; <%llvm.dbg.variable.type*> [#uses=0]
@.str9 = internal constant [6 x i8] c"count\00"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.basictype10 = internal constant %llvm.dbg.basictype.type { i32 393252, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str11, i32 0, i32 0), { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str11 = internal constant [18 x i8] c"long unsigned int\00"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram12 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str13, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str13, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 24, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype7 to { }*), i1 true, i1 true }		; <%llvm.dbg.subprogram.type*> [#uses=0]
@.str13 = internal constant [7 x i8] c"FooTwo\00"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram14 = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str15, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str15, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str16, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 29, { }* null, i1 false, i1 true }		; <%llvm.dbg.subprogram.type*> [#uses=0]
@.str15 = internal constant [9 x i8] c"FooThree\00"		; <[9 x i8]*> [#uses=1]
@.str16 = internal constant [13 x i8] c"_Z8FooThreev\00"		; <[13 x i8]*> [#uses=1]

declare void @_Z8FooThreev() nounwind

define internal i32 @_ZL10AssertFailv() nounwind {
entry:
	unreachable
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @abort() noreturn nounwind

declare void @llvm.dbg.region.end({ }*) nounwind

declare i32 @_ZL6FooOnem(i32) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare i32 @_ZL6FooTwov() nounwind
