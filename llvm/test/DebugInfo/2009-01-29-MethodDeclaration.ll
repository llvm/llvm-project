; RUN: llvm-as < %s | llc | grep 0x3C | count 1
; Check DW_AT_declaration attribute for class method foo.
target triple = "i386-apple-darwin*"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32, i8*, i8* }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i8*, i8* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, i8*, i8* }
	%llvm.dbg.global_variable.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, { }*, i8*, i8* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, i8*, i8* }
	%struct.A = type <{ i8 }>
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [6 x i8] c"cl.cc\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1 = internal constant [26 x i8] c"/Volumes/Nanpura/dbg.test\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@a = global %struct.A zeroinitializer		; <%struct.A*> [#uses=1]
@.str3 = internal constant [2 x i8] c"A\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@.str4 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5, i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite9 to { }*), i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [2 x { }*] [ { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) ], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite5 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array to { }*), i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str6 = internal constant [4 x i8] c"foo\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str7 = internal constant [12 x i8] c"_ZN1A3fooEv\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str6, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str6, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str7, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite5 to { }*), i1 false, i1 false, i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array8 = internal constant [1 x { }*] [ { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*) ], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite9 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, i64 8, i64 8, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array8 to { }*), i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str10 = internal constant [2 x i8] c"a\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type { i32 458804, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str10, i32 0, i32 0), i8* getelementptr ([2 x i8]* @.str10, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 7, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite9 to { }*), i1 false, i1 true, { }* bitcast (%struct.A* @a to { }*), i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]
