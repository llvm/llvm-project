; RUN: llvm-as < %s | llc | %prcontext ST 1 | grep 0x1 | count 1

target triple = "i386-apple-darwin9.6"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8* }
	%llvm.dbg.compositetype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
	%struct.ST = type opaque
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 393262, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* null, i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 393233, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([36 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 393216, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [4 x i8] c"t.c\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@.str1 = internal constant [36 x i8] c"/Users/echeng/LLVM/radars/r6395152/\00", section "llvm.metadata"		; <[36 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5628) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@.str3 = internal constant [4 x i8] c"foo\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 393473, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([2 x i8]* @.str4, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str4 = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 393231, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.compositetype.type* @llvm.dbg.compositetype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.compositetype = internal constant %llvm.dbg.compositetype.type { i32 393235, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([3 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 1, i64 0, i64 8, i64 0, i32 4, { }* null, { }* bitcast ([0 x { }*]* @llvm.dbg.array to { }*) }, section "llvm.metadata"		; <%llvm.dbg.compositetype.type*> [#uses=1]
@.str5 = internal constant [3 x i8] c"ST\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.array = internal constant [0 x { }*] zeroinitializer, section "llvm.metadata"		; <[0 x { }*]*> [#uses=1]

define void @foo(%struct.ST* %x1) nounwind {
entry:
	%x_addr = alloca %struct.ST*		; <%struct.ST**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%x = bitcast %struct.ST** %x_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %x, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	store %struct.ST* %x1, %struct.ST** %x_addr
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %return

return:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret void
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind
