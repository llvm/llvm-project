; Namelist is a fortran feature, this test checks whether DW_TAG_namelist and
; DW_TAG_namelist_item attributes are emitted correctly, when declared inside
; a module.
;
; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump %t.o | FileCheck %s
;
; CHECK: [[ITEM1:0x.+]]:       DW_TAG_variable
; CHECK:                          DW_AT_name  ("aa")
; CHECK: [[ITEM2:0x.+]]:       DW_TAG_variable
; CHECK:                          DW_AT_name  ("bb")
; CHECK: DW_TAG_namelist
; CHECK:    DW_AT_name  ("nml")
; CHECK: DW_TAG_namelist_item
; CHECK:    DW_AT_namelist_item ([[ITEM1]])
; CHECK: DW_TAG_namelist_item
; CHECK:    DW_AT_namelist_item ([[ITEM2]])
;
; generated from
;
; module mm
;    integer :: aa=10, bb=20
;    namelist /nml/ aa, bb
; end module mm
;
; subroutine test()
;    use mm
;    write(*,nml)
; end subroutine test
;
; Program namelist
;       Call test()
; End Program

source_filename = "namelist2.ll"

!llvm.module.flags = !{!19, !20}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "aa", scope: !2, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !4, name: "mm", file: !3, line: 1)
!3 = !DIFile(filename: "namelist2.f90", directory: "/dir")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, flags: "'+flang -g namelist2.f90 -S -emit-llvm'", runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, nameTableKind: None)
!5 = !{}
!6 = !{!0, !7, !10}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 4))
!8 = distinct !DIGlobalVariable(name: "bb", scope: !2, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "nml", scope: !2, file: !3, line: 2, type: !12, isLocal: false, isDefinition: true)
!12 = !DICompositeType(tag: DW_TAG_namelist, name: "nml", file: !3, elements: !13)
!13 = !{!1, !8}
!14 = !{!15}
!15 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !16, entity: !2, file: !3, line: 6)
!16 = distinct !DISubprogram(name: "test", scope: !4, file: !3, line: 6, type: !17, scopeLine: 6, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !14)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = distinct !DISubprogram(name: "namelist", scope: !4, file: !3, line: 11, type: !22, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!22 = !DISubroutineType(cc: DW_CC_program, types: !18)
