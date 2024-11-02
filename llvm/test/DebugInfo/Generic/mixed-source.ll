; AIX doesn't support the debug_addr section
; UNSUPPORTED:  target={{.*}}-aix{{.*}}

; RUN: %llc_dwarf -O0 -filetype=obj -o - < %s | llvm-dwarfdump -debug-line - | FileCheck %s

; CHECK: include_directories[  0] = "dir"
; CHECK-NEXT: file_names[  0]:
; CHECK-NEXT:            name: "foo.c"
; CHECK-NEXT:       dir_index: 0
; CHECK-NEXT:          source: "void foo() { }\n"
; CHECK-NEXT: file_names[  1]:
; CHECK-NEXT:            name: "bar.h"
; CHECK-NEXT:       dir_index: 0
; CHECK-NOT:           source:

; Test that DIFiles mixing source and no-source within a DICompileUnit works.

define dso_local void @foo() !dbg !5 {
  ret void, !dbg !7
}

define dso_local void @bar() !dbg !6 {
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}

!2 = !DIFile(filename: "foo.c", directory: "dir", source: "void foo() { }\0A")
!3 = !DIFile(filename: "bar.h", directory: "dir")

!4 = distinct !DICompileUnit(language: DW_LANG_C99, emissionKind: FullDebug, file: !2)
!5 = distinct !DISubprogram(name: "foo", file: !2, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !4)
!6 = distinct !DISubprogram(name: "bar", file: !3, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !4)
!7 = !DILocation(line: 1, scope: !5)
!8 = !DILocation(line: 1, scope: !6)
!9 = !DISubroutineType(types: !{})
