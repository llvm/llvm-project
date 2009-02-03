; RUN: llvm-as < %s | llc -mtriple=powerpc-apple-darwin9.5
; rdar://6499616

	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [11 x i8] c"testcase.c\00"		; <[11 x i8]*> [#uses=1]
@.str1 = internal constant [30 x i8] c"/Volumes/SandBox/NightlyTest/\00"		; <[30 x i8]*> [#uses=1]
@.str2 = internal constant [57 x i8] c"4.2.1 (Based on Apple Inc. build 5628) (LLVM build 9999)\00"		; <[57 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([11 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([57 x i8]* @.str2, i32 0, i32 0) }		; <%llvm.dbg.compile_unit.type*> [#uses=0]
@"\01LC" = internal constant [13 x i8] c"conftest.val\00"		; <[13 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
	%0 = call i8* @fopen(i8* getelementptr ([13 x i8]* @"\01LC", i32 0, i32 0), i8* null) nounwind		; <i8*> [#uses=0]
	unreachable
}

declare i8* @fopen(i8*, i8*)
