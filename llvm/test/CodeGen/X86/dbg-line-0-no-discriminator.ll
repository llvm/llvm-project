; RUN: llc -filetype=obj -use-unknown-locations=Enable -mtriple=x86_64-unknown-linux %s -o %t
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

define void @_Z3bazv() !dbg !6 {
  call void @_Z3foov(), !dbg !9
  call void @_Z3foov() ; no !dbg, so will be marked as line 0
  ret void, !dbg !11
}

declare void @_Z3foov()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 267219)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 267219)"}
!6 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 4, column: 3, scope: !10)
!10 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 1)
!11 = !DILocation(line: 6, column: 1, scope: !6)

; Look at the lengths. We can't verify the line-number-program size
; directly, but the difference in the two lengths should not change
; unexpectedly.
; CHECK:    total_length: 0x00000043
; CHECK: prologue_length: 0x0000001e
;
; Verify that we see a line entry with a discriminator, and the next entry
; has line 0 and no discriminator.
;             line column file ISA discriminator
; CHECK:      4    3      1    0   1
; CHECK-NEXT: 0    3      1    0   0
