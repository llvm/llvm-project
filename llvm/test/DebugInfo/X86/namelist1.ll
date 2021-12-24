; Namelist is a fortran feature, this test checks whether DW_TAG_namelist and
; DW_TAG_namelist_item attributes are emitted correctly.
;
; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump %t.o | FileCheck %s
;
; CHECK: [[ITEM1:0x.+]]:       DW_TAG_variable
; CHECK:                          DW_AT_name  ("a")
; CHECK: [[ITEM2:0x.+]]:       DW_TAG_variable
; CHECK:                          DW_AT_name  ("b")
; CHECK: DW_TAG_namelist
; CHECK:    DW_AT_name  ("nml")
; CHECK: DW_TAG_namelist_item
; CHECK:    DW_AT_namelist_item ([[ITEM1]])
; CHECK: DW_TAG_namelist_item
; CHECK:    DW_AT_namelist_item ([[ITEM2]])
;
; generated from
;
; program main
;
;  integer :: a=1, b
;  namelist /nml/ a, b
;
;  a = 10
;  b = 20
;  Write(*,nml)
;
; end program main

source_filename = "namelist.ll"

define void @MAIN_() !dbg !2 {
L.entry:
  %b_350 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %b_350, metadata !12, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !13
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!10, !11}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_plus_uconst, 120))
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "main", scope: !4, file: !3, line: 1, type: !7, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "namelist.f90", directory: "/dir")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, flags: "'+flang -g namelist.f90 -S -emit-llvm'", runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5, nameTableKind: None)
!5 = !{}
!6 = !{!0}
!7 = !DISubroutineType(cc: DW_CC_program, types: !8)
!8 = !{null}
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !DILocalVariable(name: "b", scope: !2, file: !3, line: 3, type: !9)
!13 = !DILocation(line: 0, scope: !2)
!14 = distinct !DILocalVariable(scope: !2, file: !3, line: 2, type: !15, flags: DIFlagArtificial)
!15 = !DICompositeType(tag: DW_TAG_namelist, name: "nml", scope: !2, file: !3, elements: !16)
!16 = !{!1, !12}
!17 = !DILocation(line: 10, column: 1, scope: !2)
