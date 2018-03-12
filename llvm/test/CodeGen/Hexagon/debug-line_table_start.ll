; RUN: llc -march=hexagon < %s | FileCheck %s

; This test case is little iffy. It checks for line_table_start,
; which in future may be completely replaced with some other label name.
; The first check is the use, the second check is for defition.

; CHECK: .Lline_table_start0
; CHECK: .Lline_table_start0:

; Function Attrs: nounwind
define i32 @f0() #0 !dbg !5 {
b0:
  %v0 = alloca i32, align 4
  store i32 0, i32* %v0, align 4
  ret i32 0, !dbg !9
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/1.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 2, column: 3, scope: !5)
