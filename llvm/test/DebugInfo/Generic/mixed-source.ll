; AIX doesn't support the debug_addr section
; UNSUPPORTED:  target={{.*}}-aix{{.*}}

; RUN: %llc_dwarf -O0 -filetype=obj -o - < %s | llvm-dwarfdump -debug-line - | FileCheck %s

; CHECK: include_directories[  0] = "dir"
; CHECK-NEXT: file_names[  0]:
; CHECK-NEXT:            name: "main.c"
; CHECK-NEXT:       dir_index: 0
; CHECK-NOT:           source:
; CHECK-NEXT: file_names[  1]:
; CHECK-NEXT:            name: "foo.c"
; CHECK-NEXT:       dir_index: 0
; CHECK-NEXT:          source: "void foo() { }\n"
; CHECK-NEXT: file_names[  2]:
; CHECK-NEXT:            name: "newline.h"
; CHECK-NEXT:       dir_index: 0
; CHECK-NEXT:          source: "\n"
; CHECK-NEXT: file_names[  3]:
; CHECK-NEXT:            name: "empty.h"
; CHECK-NEXT:       dir_index: 0
; CHECK-NEXT:          source: "\n"
; CHECK-NEXT: file_names[  4]:
; CHECK-NEXT:            name: "absent.h"
; CHECK-NEXT:       dir_index: 0
; CHECK-NOT:           source:

; Test that DIFiles mixing source and no-source within a DICompileUnit works.

define dso_local void @foo() !dbg !6 {
  ret void, !dbg !7
}

define dso_local void @newline() !dbg !9 {
  ret void, !dbg !10
}

define dso_local void @empty() !dbg !12 {
  ret void, !dbg !13
}

define dso_local void @absent() !dbg !15 {
  ret void, !dbg !16
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}

!2 = distinct !DICompileUnit(language: DW_LANG_C99, emissionKind: FullDebug, file: !4)
!3 = !DISubroutineType(types: !{})
!4 = !DIFile(filename: "main.c", directory: "dir")

!5 = !DIFile(filename: "foo.c", directory: "dir", source: "void foo() { }\0A")
!6 = distinct !DISubprogram(name: "foo", file: !5, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!7 = !DILocation(line: 1, scope: !6)

!8 = !DIFile(filename: "newline.h", directory: "dir", source: "\0A")
!9 = distinct !DISubprogram(name: "newline", file: !8, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!10 = !DILocation(line: 1, scope: !9)

!11 = !DIFile(filename: "empty.h", directory: "dir", source: "")
!12 = distinct !DISubprogram(name: "empty", file: !11, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!13 = !DILocation(line: 1, scope: !12)

!14 = !DIFile(filename: "absent.h", directory: "dir")
!15 = distinct !DISubprogram(name: "absent", file: !14, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!16 = !DILocation(line: 1, scope: !15)
