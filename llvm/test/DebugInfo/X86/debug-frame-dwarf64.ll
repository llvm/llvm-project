; This checks that .debug_frame can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf64 -force-dwarf-frame-section -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-frame %t | FileCheck %s

; CHECK:      .debug_frame contents:
; CHECK:      00000000 {{.+}} ffffffffffffffff CIE
; CHECK-NEXT:   Format:                DWARF64
; CHECK:      {{.+}} 0000000000000000 FDE cie=00000000 pc=
; CHECK-NEXT:   Format:       DWARF64

; IR generated and reduced from:
; $ cat foo.c
; void foo() { }
; $ clang -g -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo() #0 !dbg !7 {
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, column: 14, scope: !7)
