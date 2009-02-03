; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att
; PR834
; END.

target datalayout = "e-p:32:32"
target triple = "i386-unknown-freebsd6.1"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, {  }*, i8*, {  }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, {  }*, i32, i8*, i8*, i8* }
	%llvm.dbg.global_variable.type = type { i32, {  }*, {  }*, i8*, i8 *, i8*, {  }*, i32, {  }*, i1, i1, {  }* }
@x = global i32 0		; <i32*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type {
    i32 327732,
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to {  }*), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([2 x i8]* @str, i64 0, i64 0), 
    i8* getelementptr ([2 x i8]* @str, i64 0, i64 0), 
    i8* null, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 1, 
    {  }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to {  }*), 
    i1 false, 
    i1 true, 
    {  }* bitcast (i32* @x to {  }*) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 327680, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type {
    i32 327697, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to {  }*), 
    i32 4, 
    i8* getelementptr ([10 x i8]* @str1, i64 0, i64 0), 
    i8* getelementptr ([32 x i8]* @str2, i64 0, i64 0), 
    i8* getelementptr ([45 x i8]* @str3, i64 0, i64 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 327680, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@str1 = internal constant [10 x i8] c"testb.cpp\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@str2 = internal constant [32 x i8] c"/Sources/Projects/DwarfTesting/\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@str3 = internal constant [45 x i8] c"4.0.1 LLVM (Apple Computer, Inc. build 5400)\00", section "llvm.metadata"		; <[45 x i8]*> [#uses=1]
@str = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type {
    i32 327716, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([4 x i8]* @str4, i64 0, i64 0), 
    {  }* null, 
    i32 0, 
    i64 32, 
    i64 32, 
    i64 0, 
    i32 0, 
    i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@str4 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
