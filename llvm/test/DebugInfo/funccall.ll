;; RUN: llvm-as < %s | llc
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, {  }*, i8*, {  }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, {  }*, i32, i8*, i8*, i8* }
	%llvm.dbg.global_variable.type = type { i32, {  }*, {  }*, i8*, i8*, i8*, {  }*, i32, {  }*, i1, i1, {  }* }
	%llvm.dbg.subprogram.type = type { i32, {  }*, {  }*, i8*, i8*, i8*, {  }*, i32, {  }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, {  }*, i8*, {  }*, i32, {  }* }
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type {
    i32 393262, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to {  }*), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([4 x i8]* @str, i32 0, i32 0), 
    i8* getelementptr ([4 x i8]* @str, i32 0, i32 0), 
    i8* null, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 4, 
    {  }* null, 
    i1 false, 
    i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@str = internal constant [4 x i8] c"foo\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type {
    i32 393233, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to {  }*), 
    i32 1, 
    i8* getelementptr ([11 x i8]* @str1, i32 0, i32 0), 
    i8* getelementptr ([50 x i8]* @str2, i32 0, i32 0), 
    i8* getelementptr ([45 x i8]* @str3, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@str1 = internal constant [11 x i8] c"funccall.c\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@str2 = internal constant [50 x i8] c"/Volumes/Big2/llvm/llvm/test/Regression/Debugger/\00", section "llvm.metadata"		; <[50 x i8]*> [#uses=1]
@str3 = internal constant [45 x i8] c"4.0.1 LLVM (Apple Computer, Inc. build 5421)\00", section "llvm.metadata"		; <[45 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type {
    i32 393472, 
    {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to {  }*), 
    i8* getelementptr ([2 x i8]* @str4, i32 0, i32 0), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 5, 
    {  }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to {  }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@str4 = internal constant [2 x i8] c"t\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type {
    i32 393252, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([4 x i8]* @str15, i32 0, i32 0), 
    {  }* null, 
    i32 0, 
    i64 32, 
    i64 32, 
    i64 0, 
    i32 0, 
    i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@str15 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.subprogram2 = internal constant %llvm.dbg.subprogram.type {
    i32 393262, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to {  }*), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([5 x i8]* @str6, i32 0, i32 0), 
    i8* getelementptr ([5 x i8]* @str6, i32 0, i32 0), 
    i8* null, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 8, 
    {  }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to {  }*), 
    i1 false, 
    i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@str6 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable3 = internal constant %llvm.dbg.variable.type {
    i32 393474, 
    {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram2 to {  }*), 
    i8* getelementptr ([7 x i8]* @str7, i32 0, i32 0), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 8, 
    {  }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to {  }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@str7 = internal constant [7 x i8] c"retval\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type {
    i32 393268, 
    {  }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to {  }*), 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i8* getelementptr ([2 x i8]* @str4, i32 0, i32 0), 
    i8* getelementptr ([2 x i8]* @str4, i32 0, i32 0), 
    i8* null, 
    {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*), 
    i32 2, 
    {  }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to {  }*), 
    i1 true, 
    i1 true, 
    {  }* bitcast (i32* @q to {  }*) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]
@str4.upgrd.1 = internal constant [2 x i8] c"q\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=0]
@q = internal global i32 0		; <i32*> [#uses=7]

declare void @llvm.dbg.func.start({  }*)

declare void @llvm.dbg.stoppoint(i32, i32, {  }*)

declare void @llvm.dbg.declare({  }*, {  }*)

declare void @llvm.dbg.region.start({  }*)

declare void @llvm.dbg.region.end({  }*)

define void @foo() {
entry:
	%t = alloca i32, align 4		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start( {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to {  }*) )
	call void @llvm.dbg.stoppoint( i32 4, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	%t.upgrd.2 = bitcast i32* %t to {  }*		; <{  }*> [#uses=1]
	call void @llvm.dbg.declare( {  }* %t.upgrd.2, {  }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to {  }*) )
	call void @llvm.dbg.stoppoint( i32 5, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	%tmp = load i32* @q		; <i32> [#uses=1]
	store i32 %tmp, i32* %t
	call void @llvm.dbg.stoppoint( i32 6, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	%tmp1 = load i32* %t		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	store i32 %tmp2, i32* @q
	call void @llvm.dbg.stoppoint( i32 7, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	call void @llvm.dbg.region.end( {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to {  }*) )
	ret void
}

define i32 @main() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=3]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start( {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram2 to {  }*) )
	call void @llvm.dbg.stoppoint( i32 8, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	%retval.upgrd.3 = bitcast i32* %retval to {  }*		; <{  }*> [#uses=1]
	call void @llvm.dbg.declare( {  }* %retval.upgrd.3, {  }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable3 to {  }*) )
	call void @llvm.dbg.stoppoint( i32 9, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	store i32 0, i32* @q
	call void @llvm.dbg.stoppoint( i32 10, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	call void (...)* bitcast (void ()* @foo to void (...)*)( )
	call void @llvm.dbg.stoppoint( i32 11, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	%tmp.upgrd.4 = load i32* @q		; <i32> [#uses=1]
	%tmp1 = sub i32 %tmp.upgrd.4, 1		; <i32> [#uses=1]
	store i32 %tmp1, i32* @q
	call void @llvm.dbg.stoppoint( i32 13, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	%tmp2 = load i32* @q		; <i32> [#uses=1]
	store i32 %tmp2, i32* %tmp
	%tmp3 = load i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp3, i32* %retval
	%retval.upgrd.5 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint( i32 14, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to {  }*) )
	call void @llvm.dbg.region.end( {  }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram2 to {  }*) )
	ret i32 %retval.upgrd.5
}
