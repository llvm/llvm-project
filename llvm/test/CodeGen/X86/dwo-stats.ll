; REQUIRES: asserts
; RUN: llc %s -mtriple=x86_64-linux --split-dwarf-file=%t.dwo --split-dwarf-output=%t.dwo --filetype=obj -o /dev/null -stats 2>&1 | FileCheck %s --check-prefixes=SPLIT,CHECK
; RUN: llc %s -mtriple=x86_64-linux --filetype=obj -o /dev/null -stats 2>&1 | FileCheck %s --check-prefixes=NOTSPLIT,CHECK

; NOTSPLIT-NOT: {{[0-9]+}} elf-object-writer - Total size of sections written to .dwo file
; CHECK-DAG: {{[0-9]+}} elf-object-writer - Total size of debug info sections
; SPLIT-DAG: {{[0-9]+}} elf-object-writer - Total size of sections written to .dwo file
; NOTSPLIT-NOT: {{[0-9]+}} elf-object-writer - Total size of sections written to .dwo file

define void @banana() !dbg !8 {
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.1", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "test.dwo", emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.1"}
!8 = distinct !DISubprogram(name: "banana", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "test.c", directory: "/tmp")
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 1, column: 20, scope: !8)
