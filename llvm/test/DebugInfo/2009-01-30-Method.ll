; RUN: llvm-as < %s | llc | grep "\\"foo" | count 3
; 1 declaration, 1 definition and 1 pubnames entry.
target triple = "i386-apple-darwin*"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
	%struct.Fibonancci = type { i32 }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [10 x i8] c"method.cc\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str1 = internal constant [64 x i8] c"/Volumes/Nanpura/mainline/llvmgcc42.build/gcc/../../../dbg.test\00", section "llvm.metadata"		; <[64 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([10 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([64 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [11 x i8] c"Fibonancci\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str4 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str5 = internal constant [2 x i8] c"N\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 5, i64 32, i64 32, i64 0, i32 1, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype6 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype6 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite7 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str8 = internal constant [4 x i8] c"foo\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str9 = internal constant [22 x i8] c"_ZN10Fibonancci3fooEi\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str9, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 10, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite7 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array10 = internal constant [2 x { }*] [ { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*) ], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite11 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array10 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype12 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array13 = internal constant [3 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array13 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram14 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str9, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 10, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype15 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str16 = internal constant [5 x i8] c"this\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram14 to { }*), i8* getelementptr ([5 x i8]* @.str16, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 10, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype15 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str17 = internal constant [2 x i8] c"i\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable18 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram14 to { }*), i8* getelementptr ([2 x i8]* @.str17, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 10, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.array19 = internal constant [1 x { }*] [ { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) ], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite20 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array19 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str21 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram22 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str21, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str21, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 14, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite20 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str23 = internal constant [4 x i8] c"fib\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.variable24 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram22 to { }*), i8* getelementptr ([4 x i8]* @.str23, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 15, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]

define void @_ZN10Fibonancci3fooEi(%struct.Fibonancci* %this, i32 %i) nounwind {
entry:
	%this_addr = alloca %struct.Fibonancci*		; <%struct.Fibonancci**> [#uses=3]
	%i_addr = alloca i32		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram14 to { }*))
	%0 = bitcast %struct.Fibonancci** %this_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	store %struct.Fibonancci* %this, %struct.Fibonancci** %this_addr
	%1 = bitcast i32* %i_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable18 to { }*))
	store i32 %i, i32* %i_addr
	call void @llvm.dbg.stoppoint(i32 11, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = load %struct.Fibonancci** %this_addr, align 4		; <%struct.Fibonancci*> [#uses=1]
	%3 = getelementptr %struct.Fibonancci* %2, i32 0, i32 0		; <i32*> [#uses=1]
	%4 = load i32* %i_addr, align 4		; <i32> [#uses=1]
	store i32 %4, i32* %3, align 4
	call void @llvm.dbg.stoppoint(i32 12, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %return

return:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 12, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram14 to { }*))
	ret void
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind

define i32 @main() nounwind {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%fib = alloca %struct.Fibonancci		; <%struct.Fibonancci*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram22 to { }*))
	%1 = bitcast %struct.Fibonancci* %fib to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable24 to { }*))
	call void @llvm.dbg.stoppoint(i32 16, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @_ZN10Fibonancci3fooEi(%struct.Fibonancci* %fib, i32 42) nounwind
	call void @llvm.dbg.stoppoint(i32 17, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 0, i32* %0, align 4
	%2 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %2, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 17, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram22 to { }*))
	ret i32 %retval1
}
