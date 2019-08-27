; RUN: llvm-as %s -disable-verify -o %t.bc
; RUN: llvm-lto -exported-symbol f -exported-symbol _f \
; RUN:     -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-WARN
; RUN: llvm-nm %t.o | FileCheck %s 

; Check that missing debug locations on inlinable calls are a
; recoverable error.

; CHECK-WARN: warning{{.*}} ignoring invalid debug info
; CHECK: {{f$}}
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

define void @h() #0 !dbg !7 {
entry:
  call void (...) @i(), !dbg !9
  ret void, !dbg !10
}

declare void @i(...) #1

define void @g() #0 !dbg !11 {
entry:
; Manually removed !dbg.
  call void @h()
  ret void, !dbg !13
}

define void @f() #0 !dbg !14 {
entry:
  call void @g(), !dbg !15
  ret void, !dbg !16
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 12, scope: !7)
!10 = !DILocation(line: 2, column: 17, scope: !7)
!11 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 3, column: 12, scope: !11)
!13 = !DILocation(line: 3, column: 17, scope: !11)
!14 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: false, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 4, column: 12, scope: !14)
!16 = !DILocation(line: 4, column: 17, scope: !14)
