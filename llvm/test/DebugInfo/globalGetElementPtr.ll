; RUN: llvm-as < %s | llc
; ModuleID = 'foo.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, %struct.anon*, i8*, %struct.anon*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, %struct.anon*, i32, i8*, i8*, i8* }
	%llvm.dbg.compositetype.type = type { i32, %struct.anon*, i8*, %struct.anon*, i32, i64, i64, i64, i32, %struct.anon*, %struct.anon* }
	%llvm.dbg.derivedtype.type = type { i32, %struct.anon*, i8*, %struct.anon*, i32, i64, i64, i64, i32, %struct.anon* }
	%llvm.dbg.global_variable.type = type { i32, %struct.anon*, %struct.anon*, i8*, i8*, i8*, %struct.anon*, i32, %struct.anon*, i1, i1, %struct.anon* }
	%llvm.dbg.subprogram.type = type { i32, %struct.anon*, %struct.anon*, i8*, i8*, i8*, %struct.anon*, i32, %struct.anon*, i1, i1 }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%llvm.dbg.variable.type = type { i32, %struct.anon*, i8*, %struct.anon*, i32, %struct.anon* }
	%struct.S271 = type { [0 x %struct.anon], %struct.anon }
	%struct.anon = type {  }
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type {
    i32 393262, 
    %struct.anon* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %struct.anon*), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), 
    i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), 
    i8* null, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 2, 
    %struct.anon* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to %struct.anon*), 
    i1 false, 
    i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type {
    i32 393233, 
    %struct.anon* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %struct.anon*), 
    i32 1, 
    i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0), 
    i8* getelementptr ([23 x i8]* @.str1, i32 0, i32 0), 
    i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [6 x i8] c"foo.c\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1 = internal constant [23 x i8] c"/Volumes/MacOS9/tests/\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5546) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@.str3 = internal constant [4 x i8] c"var\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type {
    i32 393231, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* null, 
    %struct.anon* null, 
    i32 0, 
    i64 32, 
    i64 32, 
    i64 0, 
    i32 0, 
    %struct.anon* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type {
    i32 393252, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), 
    %struct.anon* null, 
    i32 0, 
    i64 8, 
    i64 8, 
    i64 0, 
    i32 0, 
    i32 6 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str4 = internal constant [5 x i8] c"char\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type {
    i32 393474, 
    %struct.anon* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %struct.anon*), 
    i8* getelementptr ([7 x i8]* @.str5, i32 0, i32 0), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 2, 
    %struct.anon* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str5 = internal constant [7 x i8] c"retval\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@a271 = weak global [0 x %struct.S271] zeroinitializer		; <[0 x %struct.S271]*> [#uses=3]
@llvm.dbg.subprogram6 = internal constant %llvm.dbg.subprogram.type {
    i32 393262, 
    %struct.anon* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %struct.anon*), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* getelementptr ([5 x i8]* @.str7, i32 0, i32 0), 
    i8* getelementptr ([5 x i8]* @.str7, i32 0, i32 0), 
    i8* null, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 3, 
    %struct.anon* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype8 to %struct.anon*), 
    i1 false, 
    i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str7 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.basictype8 = internal constant %llvm.dbg.basictype.type {
    i32 393252, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* getelementptr ([4 x i8]* @.str9, i32 0, i32 0), 
    %struct.anon* null, 
    i32 0, 
    i64 32, 
    i64 32, 
    i64 0, 
    i32 0, 
    i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str9 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.variable10 = internal constant %llvm.dbg.variable.type {
    i32 393474, 
    %struct.anon* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram6 to %struct.anon*), 
    i8* getelementptr ([7 x i8]* @.str5, i32 0, i32 0), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 3, 
    %struct.anon* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype8 to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type {
    i32 393268, 
    %struct.anon* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to %struct.anon*), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* getelementptr ([5 x i8]* @.str11, i32 0, i32 0), 
    i8* getelementptr ([5 x i8]* @.str11, i32 0, i32 0), 
    i8* null, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 1, 
    %struct.anon* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype to %struct.anon*), 
    i1 false, 
    i1 true, 
    %struct.anon* getelementptr ([0 x %struct.S271]* @a271, i32 0, i32 0, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str11 = internal constant [5 x i8] c"a271\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.compositetype = internal constant %llvm.dbg.compositetype.type {
    i32 393217, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* null, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 0, 
    i64 0, 
    i64 8, 
    i64 0, 
    i32 0, 
    %struct.anon* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype12 to %struct.anon*), 
    %struct.anon* bitcast ([1 x %struct.anon*]* @llvm.dbg.array25 to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.compositetype12 = internal constant %llvm.dbg.compositetype.type {
    i32 393235, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* getelementptr ([5 x i8]* @.str13, i32 0, i32 0), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 1, 
    i64 0, 
    i64 8, 
    i64 0, 
    i32 0, 
    %struct.anon* null, 
    %struct.anon* bitcast ([2 x %struct.anon*]* @llvm.dbg.array23 to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@.str13 = internal constant [5 x i8] c"S271\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype14 = internal constant %llvm.dbg.derivedtype.type {
    i32 393229, 
    %struct.anon* null, 
    i8* getelementptr ([2 x i8]* @.str15, i32 0, i32 0), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 1, 
    i64 0, 
    i64 8, 
    i64 0, 
    i32 0, 
    %struct.anon* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype16 to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str15 = internal constant [2 x i8] c"a\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.compositetype16 = internal constant %llvm.dbg.compositetype.type {
    i32 393217, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* null, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 0, 
    i64 0, 
    i64 8, 
    i64 0, 
    i32 0, 
    %struct.anon* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype17 to %struct.anon*), 
    %struct.anon* bitcast ([1 x %struct.anon*]* @llvm.dbg.array18 to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.compositetype17 = internal constant %llvm.dbg.compositetype.type {
    i32 393235, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* null, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 1, 
    i64 0, 
    i64 8, 
    i64 0, 
    i32 0, 
    %struct.anon* null, 
    %struct.anon* bitcast ([0 x %struct.anon*]* @llvm.dbg.array to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.array = internal constant [0 x %struct.anon*] zeroinitializer, section "llvm.metadata"		; <[0 x %struct.anon*]*> [#uses=1]
@llvm.dbg.subrange = internal constant %llvm.dbg.subrange.type {
    i32 393249, 
    i64 0, 
    i64 4 }, section "llvm.metadata"		; <%llvm.dbg.subrange.type*> [#uses=1]
@llvm.dbg.array18 = internal constant [1 x %struct.anon*] [ %struct.anon* bitcast (%llvm.dbg.subrange.type* @llvm.dbg.subrange to %struct.anon*) ], section "llvm.metadata"		; <[1 x %struct.anon*]*> [#uses=1]
@llvm.dbg.derivedtype19 = internal constant %llvm.dbg.derivedtype.type {
    i32 393229, 
    %struct.anon* null, 
    i8* getelementptr ([2 x i8]* @.str20, i32 0, i32 0), 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 1, 
    i64 0, 
    i64 8, 
    i64 0, 
    i32 0, 
    %struct.anon* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype21 to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str20 = internal constant [2 x i8] c"b\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.compositetype21 = internal constant %llvm.dbg.compositetype.type {
    i32 393235, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i8* null, 
    %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*), 
    i32 1, 
    i64 0, 
    i64 8, 
    i64 0, 
    i32 0, 
    %struct.anon* null, 
    %struct.anon* bitcast ([0 x %struct.anon*]* @llvm.dbg.array22 to %struct.anon*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@llvm.dbg.array22 = internal constant [0 x %struct.anon*] zeroinitializer, section "llvm.metadata"		; <[0 x %struct.anon*]*> [#uses=1]
@llvm.dbg.array23 = internal constant [2 x %struct.anon*] [ %struct.anon* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype14 to %struct.anon*), %struct.anon* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to %struct.anon*) ], section "llvm.metadata"		; <[2 x %struct.anon*]*> [#uses=1]
@llvm.dbg.subrange24 = internal constant %llvm.dbg.subrange.type {
    i32 393249, 
    i64 0, 
    i64 4 }, section "llvm.metadata"		; <%llvm.dbg.subrange.type*> [#uses=1]
@llvm.dbg.array25 = internal constant [1 x %struct.anon*] [ %struct.anon* bitcast (%llvm.dbg.subrange.type* @llvm.dbg.subrange24 to %struct.anon*) ], section "llvm.metadata"		; <[1 x %struct.anon*]*> [#uses=1]

define i8* @var() {
entry:
	%retval = alloca i8*		; <i8**> [#uses=3]
	%tmp = alloca i8*		; <i8**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start( %struct.anon* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %struct.anon*) )
	call void @llvm.dbg.stoppoint( i32 2, i32 0, %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*) )
	%retval1 = bitcast i8** %retval to %struct.anon*		; <%struct.anon*> [#uses=1]
	call void @llvm.dbg.declare( %struct.anon* %retval1, %struct.anon* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to %struct.anon*) )
	bitcast %struct.S271* getelementptr ([0 x %struct.S271]* @a271, i32 0, i32 0) to i8*		; <i8*>:0 [#uses=0]
	store i8* bitcast ([0 x %struct.S271]* @a271 to i8*), i8** %tmp, align 4
	%tmp2 = load i8** %tmp, align 4		; <i8*> [#uses=1]
	store i8* %tmp2, i8** %retval, align 4
	br label %return

return:		; preds = %entry
	%retval3 = load i8** %retval		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint( i32 2, i32 0, %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*) )
	call void @llvm.dbg.region.end( %struct.anon* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %struct.anon*) )
	ret i8* %retval3
}

declare void @llvm.dbg.func.start(%struct.anon*) nounwind 

declare void @llvm.dbg.stoppoint(i32, i32, %struct.anon*) nounwind 

declare void @llvm.dbg.declare(%struct.anon*, %struct.anon*) nounwind 

declare void @llvm.dbg.region.end(%struct.anon*) nounwind 

define i32 @main() {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start( %struct.anon* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram6 to %struct.anon*) )
	call void @llvm.dbg.stoppoint( i32 3, i32 0, %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*) )
	%retval1 = bitcast i32* %retval to %struct.anon*		; <%struct.anon*> [#uses=1]
	call void @llvm.dbg.declare( %struct.anon* %retval1, %struct.anon* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable10 to %struct.anon*) )
	br label %return

return:		; preds = %entry
	%retval2 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint( i32 3, i32 0, %struct.anon* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %struct.anon*) )
	call void @llvm.dbg.region.end( %struct.anon* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram6 to %struct.anon*) )
	ret i32 %retval2
}
