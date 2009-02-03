; RUN: llvm-as < %s | llc | grep DWARF
; ModuleID = 'foo.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, {  }*, i8*, {  }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, {  }*, i32, i8*, i8*, i8* }
	%llvm.dbg.global_variable.type = type { i32, {  }*, {  }*, i8*, i8*, i8*, {  }*, i32, {  }*, i1, i1, {  }* }
@x = common global i32 0		; <i32*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type {
    i32 393268, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to {  }*), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([2 x i8]* @.str3, i32 0, i32 0), 
    i8* getelementptr ([2 x i8]* @.str3, i32 0, i32 0), 
    i8* null, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 1, 
    {  }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to {  }*), 
    i1 false, 
    i1 true, 
    {  }* bitcast (i32* @x to {  }*) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type {
    i32 393233, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to {  }*), 
    i32 1, 
    i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0), 
    i8* getelementptr ([23 x i8]* @.str1, i32 0, i32 0), 
    i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [6 x i8] c"foo.c\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1 = internal constant [23 x i8] c"/Volumes/MacOS9/tests/\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5555) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@.str3 = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type {
    i32 393252, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), 
    {  }* null, 
    i32 0, 
    i64 32, 
    i64 32, 
    i64 0, 
    i32 0, 
    i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str4 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
