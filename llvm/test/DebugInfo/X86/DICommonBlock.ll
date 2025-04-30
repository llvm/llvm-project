; ModuleID = 'none.f90'
; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; CHECK: DW_TAG_common_block

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

@common_a = common global [32 x i8] zeroinitializer, align 8, !dbg !13

define i32 @subr() !dbg !9 {
    %1 = getelementptr inbounds [32 x i8], [32 x i8]* @common_a, i64 0, i32 8
    %2 = bitcast i8* %1 to i32*
    %3 = load i32, i32* %2
    ret i32 %3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !1, producer: "PGI Fortran", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, retainedTypes: !14, globals: !2)
!1 = !DIFile(filename: "none.f90", directory: "/not/here/")
!2 = !{!13}
!3 = !{}
!4 = !DIGlobalVariable(name: "common /a/", scope: !5, file: !1, line: 4, isLocal: false, isDefinition: true, type: !12)
!5 = !DICommonBlock(scope: !9, declaration: !4, name: "a", file: !1, line: 4)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"PGI Fortran"}
!9 = distinct !DISubprogram(name: "subrtn", scope: !0, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32)
!13 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!14 = !{!12, !10}
