; PR2614
; RUN: llvm-as < %s | llc | grep "label" | count 8

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-f80:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, {  }*, i8*, {  }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, {  }*, i32, i8*, i8*, i8* }
	%llvm.dbg.compositetype.type = type { i32, {  }*, i8*, {  }*, i32, i64, i64, i64, i32, {  }*, {  }* }
	%llvm.dbg.derivedtype.type = type { i32, {  }*, i8*, {  }*, i32, i64, i64, i64, i32, {  }* }
	%llvm.dbg.global_variable.type = type { i32, {  }*, {  }*, i8*, i8*, i8*, {  }*, i32, {  }*, i1, i1, {  }* }
	%llvm.dbg.subprogram.type = type { i32, {  }*, {  }*, i8*, i8*, i8*, {  }*, i32, {  }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, {  }*, i8*, {  }*, i32, {  }* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=0]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [17 x i8] c"deaddebuglabel.d\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@.str1 = internal constant [50 x i8] c"/home/kamm/eigenes/projekte/llvmdc/llvmdc/mytests\00", section "llvm.metadata"		; <[50 x i8]*> [#uses=1]
@.str2 = internal constant [48 x i8] c"LLVMDC (http://www.dsource.org/projects/llvmdc)\00", section "llvm.metadata"		; <[48 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type {
    i32 393233, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to {  }*), 
    i32 2, 
    i8* getelementptr ([17 x i8]* @.str, i32 0, i32 0), 
    i8* getelementptr ([50 x i8]* @.str1, i32 0, i32 0), 
    i8* getelementptr ([48 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str5 = internal constant [20 x i8] c"deaddebuglabel.main\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@.str6 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram7 = internal constant %llvm.dbg.subprogram.type {
    i32 393262, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to {  }*), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([20 x i8]* @.str5, i32 0, i32 0), 
    i8* getelementptr ([20 x i8]* @.str5, i32 0, i32 0), 
    i8* getelementptr ([5 x i8]* @.str6, i32 0, i32 0), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 3, 
    {  }* null, 
    i1 false, 
    i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]

declare void @llvm.dbg.func.start({  }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, {  }*) nounwind

declare void @llvm.dbg.region.end({  }*) nounwind

define fastcc i32 @main() {
entry.main:
	call void @llvm.dbg.func.start( {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to {  }*) )
	br i1 true, label %reachable, label %unreachable

reachable:		; preds = %entry.main
	call void @llvm.dbg.region.end( {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to {  }*) )
	ret i32 1

unreachable:		; preds = %entry.main
	call void @llvm.dbg.stoppoint( i32 7, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	call void @llvm.dbg.region.end( {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to {  }*) )
	ret i32 0
}
