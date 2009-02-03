; RUN: llvm-as < %s | opt -mem2reg | llvm-dis | grep alloca | count 1
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 393233, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([7 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([55 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [7 x i8] c"adce.c\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1 = internal constant [6 x i8] c"/tmp/\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str2 = internal constant [55 x i8] c"4.2.1 (Based on Apple Inc. build 5623) (LLVM build 00)\00", section "llvm.metadata"		; <[55 x i8]*> [#uses=1]
@.str3 = internal constant [4 x i8] c"foo\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 393252, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str4 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 393472, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([2 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str5 = internal constant [2 x i8] c"i\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]

define i32 @foo() nounwind {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%i = alloca i32		; <i32*> [#uses=4]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%i1 = bitcast i32* %i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %i1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	call void @llvm.dbg.stoppoint(i32 3, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 4, i32* %i, align 4
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%1 = load i32* %i, align 4		; <i32> [#uses=1]
	%2 = mul i32 %1, 84		; <i32> [#uses=1]
	store i32 %2, i32* %i, align 4
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 42, i32* %0, align 4
	%3 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %3, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval2 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret i32 %retval2
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind
