; REQUIRES: x86_64-linux
; Checks if symbolizer can correctly symbolize address in the padding between
; functions.
; RUN: llc  -o %t.o -filetype=obj -mtriple=x86_64-pc-linux  %s
; RUN: echo 0x5 | llvm-symbolizer -obj=%t.o | FileCheck %s --check-prefix=FOO
; RUN: echo 0xd | llvm-symbolizer -obj=%t.o | FileCheck %s --check-prefix=PADDING
; RUN: echo 0x10 | llvm-symbolizer -obj=%t.o | FileCheck %s --check-prefix=MAIN

;FOO: foo
;PADDING: ??
;MAIN: main

@a = global i32 1, align 4

define i32 @foo() !dbg !9 {
entry:
  %0 = load i32, i32* @a, align 4
  ret i32 %0
}

define i32 @main() !dbg !14 {
entry:
  %call = call i32 @foo(), !dbg !18
  ret i32 %call
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "padding-x86_64.c", directory: "/tmp/")
!2 = !{}
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!5}
!14 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !10, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: false, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 7, column: 8, scope: !14)
