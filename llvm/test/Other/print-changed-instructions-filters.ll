; RUN: opt -passes=globaldce -disable-output -print-changed=inst-quiet -filter-print-source-locs=missing.c:1 %s 2>&1 | FileCheck %s --allow-empty --check-prefix=EMPTY
; RUN: opt -passes=globaldce -disable-output -print-changed=inst-quiet -filter-print-funcs=missing -print-module-scope %s 2>&1 | FileCheck %s --allow-empty --check-prefix=EMPTY
; RUN: opt -passes=globaldce -disable-output -print-changed=inst-quiet -filter-print-source-locs=source.c:1 %s 2>&1 | FileCheck %s --check-prefix=REMOVED

define internal i32 @drop() !dbg !5 {
entry:
  ret i32 0, !dbg !8
}

define i32 @keep() !dbg !6 {
entry:
  ret i32 1, !dbg !9
}

; EMPTY-NOT: IR Instruction Changes

; REMOVED:      *** IR Instruction Changes After GlobalDCEPass on [module] ***
; REMOVED-NEXT: - block#[[BLOCK:[0-9]+]] @drop:0
; REMOVED-NEXT: - inst#[[INST:[0-9]+]] @drop block#[[BLOCK]]:0   ret i32 0, !dbg !{{[0-9]+}}
; REMOVED-NEXT: ; summary: instructions +0 -1 changed 0 moved 0; blocks +0 -1 moved 0

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "source.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !7)
!5 = distinct !DISubprogram(name: "drop", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = distinct !DISubprogram(name: "keep", scope: !1, file: !1, line: 2, type: !4, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!7 = !{!10}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !6)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
