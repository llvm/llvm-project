; RUN: llvm-as < %s | llc | grep 0x49 | count 3
; Count number of DW_AT_Type attributes.
target triple = "i386-apple-darwin*"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }* }
	%llvm.dbg.global_variable.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, { }* }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [8 x i8] c"array.c\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1 = internal constant [26 x i8] c"/Volumes/Nanpura/dbg.test\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([8 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@c = common global [3 x i32] zeroinitializer		; <[3 x i32]*> [#uses=1]
@llvm.dbg.subrange = internal constant %llvm.dbg.subrange.type { i32 458785, i64 0, i64 2 }, section "llvm.metadata"		; <%llvm.dbg.subrange.type*> [#uses=1]
@llvm.dbg.array = internal constant [1 x { }*] [ { }* bitcast (%llvm.dbg.subrange.type* @llvm.dbg.subrange to { }*) ], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@.str3 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458753, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 96, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast ([1 x { }*]* @llvm.dbg.array to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str4 = internal constant [2 x i8] c"c\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type { i32 458804, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([2 x i8]* @.str4, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 false, i1 true, { }* bitcast ([3 x i32]* @c to { }*) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]
