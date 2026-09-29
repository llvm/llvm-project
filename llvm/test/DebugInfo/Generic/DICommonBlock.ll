; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; CHECK: DW_TAG_common_block
; CHECK-DAG: DW_AT_name{{.*}}"a"
; CHECK-DAG: DW_AT_location
; CHECK: DW_TAG_variable
; CHECK-DAG: DW_AT_name{{.*}}"c"
; CHECK-DAG: DW_AT_location{{.*}}DW_OP_plus_uconst{{.*}}4
; CHECK: {{DW_TAG|NULL}}

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@common_a = common global [32 x i8] zeroinitializer, align 8, !dbg !13, !dbg !15

define i32 @subr() !dbg !9 {
    %1 = getelementptr inbounds [32 x i8], ptr @common_a, i64 0, i32 8
    %2 = load i32, ptr %1
    ret i32 %2
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !1, producer: "PGI Fortran", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, retainedTypes: !14, globals: !3)
!1 = !DIFile(filename: "none.f90", directory: "/not/here/")
!2 = distinct !DIGlobalVariable(scope: !5, name: "c", file: !1, type: !12, isDefinition: true)
!3 = !{!13, !15}
!4 = distinct !DIGlobalVariable(scope: !5, name: "COMMON /foo/", file: !1, line: 4, isLocal: false, isDefinition: true, type: !12)
!5 = !DICommonBlock(scope: !9, declaration: !4, name: "a", file: !1, line: 4)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"PGI Fortran"}
!9 = distinct !DISubprogram(name: "s", scope: !0, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32)
!13 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!14 = !{!12, !10}
!15 = !DIGlobalVariableExpression(var: !2, expr: !DIExpression(DW_OP_plus_uconst, 4))
