; Test for DIFlagPure, DIFlagElement and DIFlagRecursive. These three
; DIFlags are used to attach DW_AT_pure, DW_AT_element, and DW_AT_recursive
; attributes to DW_TAG_subprogram DIEs.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; CHECK: !DISubprogram({{.*}}, spFlags: DISPFlagDefinition | DISPFlagPure | DISPFlagElemental | DISPFlagRecursive,

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

define void @subprgm() !dbg !6 {
L:
  ret void
}

!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: "Flang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "fortran-subprogram-at.f", directory: "/")
!4 = !{}
!5 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!6 = distinct !DISubprogram(name: "subprgm", scope: !2, file: !3, line: 256, type: !7, scopeLine: 256, spFlags: DISPFlagDefinition | DISPFlagPure | DISPFlagElemental | DISPFlagRecursive, unit: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !5}
