; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

@common_a = common global [32 x i8] zeroinitializer, align 8, !dbg !13, !dbg !15

define i32 @subr() !dbg !9 {
    %1 = getelementptr inbounds [32 x i8], ptr @common_a, i64 0, i32 8
    %2 = load i32, ptr %1
    ret i32 %2
}

; CHECK-DAG: ![[CU:[0-9]+]] = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: ![[FILE:[0-9]+]], producer: "PGI Fortran", isOptimized: false, runtimeVersion: 2, emissionKind: 1, retainedTypes: !{{.*}}, subprograms: ![[SUBPROG_LIST:[0-9]+]], globals: ![[GV_LIST:[0-9]+]])
; CHECK-DAG: ![[SUBPROG_LIST]] = !{![[SUBPROG:[0-9]+]]}
; CHECK-DAG: ![[SUBPROG]] = !DISubprogram(name: "s", scope: ![[CU]], file: ![[FILE]], line: 1, type: !4, isLocal: false, isDefinition: true, isOptimized: false, function: i32 ()* @subr)
; CHECK-DAG: ![[GV_LIST]] = !{![[GV1:[0-9]+]], ![[GV2:[0-9]+]]}
; CHECK-DAG: ![[GV1]] = distinct !DIGlobalVariable(name: "COMMON /foo/", scope: ![[SUBPROG]], file: ![[FILE]], line: 4, type: ![[TYPE_INT_ARRAY:[0-9]+]], isLocal: false, isDefinition: true, variable: [32 x i8]* @common_a)
; CHECK-DAG: ![[GV2]] = distinct !DIGlobalVariable(name: "c", scope: ![[SUBPROG]], file: ![[FILE]], type: ![[TYPE_INT_ARRAY]], isLocal: false, isDefinition: true)
; CHECK-DAG: ![[TYPE_INT_ARRAY]] = !DIBasicType(name: "int", size: 32)

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
