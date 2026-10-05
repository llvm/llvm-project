; RUN: rm -rf %t && split-file %s %t

; RUN: %llc_dwarf -mtriple=arm64-apple-darwin -filetype=obj %t/input.ll -o %t/input.o

; RUN: dsymutil -f --linker=classic -oso-prepend-path=%t -y %t/input.map -o %t/classic.dwarf
; RUN: llvm-dwarfdump --show-parents --name=child %t/classic.dwarf | FileCheck %s --implicit-check-not=DW_TAG_

; RUN: dsymutil -f --linker=parallel -oso-prepend-path=%t -y %t/input.map -o %t/parallel.dwarf
; RUN: llvm-dwarfdump --show-parents --name=child %t/parallel.dwarf | FileCheck %s --implicit-check-not=DW_TAG_

; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_AT_low_pc
; CHECK: DW_AT_linkage_name ("parent")
; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_low_pc (0x0000000000001004)
; CHECK-NEXT: DW_AT_high_pc (0x0000000000001008)
; CHECK: DW_AT_linkage_name ("child")

;--- input.ll
target triple = "arm64-apple-darwin"

define internal void @parent() noinline optnone !dbg !5 {
entry:
  ret void, !dbg !7
}

define void @child() noinline optnone !dbg !8 {
entry:
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "skip-pc-nested", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "skip-pc-nested.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "parent", linkageName: "parent", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !10)
!7 = !DILocation(line: 2, column: 1, scope: !5)
!8 = distinct !DISubprogram(name: "child", linkageName: "child", scope: !5, file: !1, line: 4, type: !6, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocation(line: 5, column: 1, scope: !8)
!10 = !{null}

;--- input.map
# @parent is omitted from the input map, so it will not have DW_AT_low_pc
# @child is in the input map, so it should have DW_AT_low_pc
---
triple: 'arm64-apple-darwin'
objects:
  - filename: input.o
    symbols:
      - { sym: _child, objAddr: 0x4, binAddr: 0x1004, size: 0x4 }
...
