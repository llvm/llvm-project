; RUN: opt -S -strip-debug < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; CHECK-LABEL: _Z5test1v
; CHECK-NOT: br {{.*}} !llvm.loop
define void @_Z5test1v() !dbg !7 {
entry:
  br label %while.body, !dbg !9

while.body:
  call void @_Z3barv(), !dbg !10
  br label %while.body, !dbg !11, !llvm.loop !13

return:
  ret void, !dbg !14
}

declare void @_Z3barv()

; CHECK-LABEL: _Z5test2v
; CHECK: br {{.*}} !llvm.loop [[LOOP:![0-9]+]]
define void @_Z5test2v() !dbg !15 {
entry:
  br label %while.body, !dbg !16

while.body:
  call void @_Z3barv(), !dbg !17
  br label %while.body, !dbg !18, !llvm.loop !19

return:
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0"}
!7 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, column: 3, scope: !7)
!10 = !DILocation(line: 5, column: 5, scope: !7)
!11 = !DILocation(line: 4, column: 3, scope: !12)
!12 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 1)
!13 = distinct !{!13, !9}
!14 = !DILocation(line: 6, column: 1, scope: !7)
!15 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!16 = !DILocation(line: 8, column: 14, scope: !15)
!17 = !DILocation(line: 11, column: 5, scope: !15)
!18 = !DILocation(line: 10, column: 3, scope: !15)
!19 = distinct !{!19, !16, !20}
!20 = !{!"llvm.loop.unroll.enable"}
!21 = !DILocation(line: 12, column: 1, scope: !15)

; CHECK-NOT: !DICompileUnit
; CHECK-NOT: !DIFile
; CHECK-NOT: !DISubprogram
; CHECK-NOT: !DISubroutineType
; CHECK-NOT: !DILocation
; CHECK-NOT: !DILexicalBlockFile
; CHECK: [[LOOP]] = distinct !{[[LOOP]], [[LOOP_UNROLL:![0-9]+]]}
; CHECK-NEXT: [[LOOP_UNROLL]] = !{!"llvm.loop.unroll.enable"}
; CHECK-NOT: !DILocation
