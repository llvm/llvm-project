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
declare i1 @_Z3bazv()

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

; CHECK-LABEL: _Z5test3v
define void @_Z5test3v() !dbg !22 {
entry:
  br label %while.body, !dbg !23

while.body:
  %c = call i1 @_Z3bazv()
  br i1 %c, label %if, label %then

if:
  call void @_Z3barv(), !dbg !24
; CHECK: br {{.*}} !llvm.loop [[LOOP2:![0-9]+]]
  br label %while.body, !dbg !25, !llvm.loop !26

then:
; CHECK: br {{.*}} !llvm.loop [[LOOP2]]
  br label %while.body, !dbg !25, !llvm.loop !26

return:
  ret void, !dbg !28
}

; CHECK-LABEL: _Z5test4v
; CHECK-NOT: br {{.*}} !llvm.loop
define void @_Z5test4v() !dbg !30 {
entry:
  br label %while.body, !dbg !31

while.body:
  call void @_Z3barv(), !dbg !32
  br label %while.body, !dbg !33, !llvm.loop !34

return:
  ret void, !dbg !36
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
!22 = distinct !DISubprogram(name: "test3", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!23 = !DILocation(line: 8, column: 14, scope: !22)
!24 = !DILocation(line: 11, column: 5, scope: !22)
!25 = !DILocation(line: 10, column: 3, scope: !22)
!26 = distinct !{!26, !23, !29, !27}
!27 = !{!"llvm.loop.unroll.enable"}
!28 = !DILocation(line: 12, column: 1, scope: !22)
!29 = !DILocation(line: 12, column: 1, scope: !22)
!30 = distinct !DISubprogram(name: "test4", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!31 = !DILocation(line: 8, column: 14, scope: !30)
!32 = !DILocation(line: 11, column: 5, scope: !30)
!33 = !DILocation(line: 10, column: 3, scope: !30)
!34 = distinct !{!34, !31, !35}
!35 = !DILocation(line: 12, column: 1, scope: !30)
!36 = !DILocation(line: 12, column: 1, scope: !30)

; CHECK-NOT: !DICompileUnit
; CHECK-NOT: !DIFile
; CHECK-NOT: !DISubprogram
; CHECK-NOT: !DISubroutineType
; CHECK-NOT: !DILocation
; CHECK-NOT: !DILexicalBlockFile
; CHECK: [[LOOP]] = distinct !{[[LOOP]], [[LOOP_UNROLL:![0-9]+]]}
; CHECK-NEXT: [[LOOP_UNROLL]] = !{!"llvm.loop.unroll.enable"}
; CHECK: [[LOOP2]] = distinct !{[[LOOP2]], [[LOOP_UNROLL]]}
; CHECK-NOT: !DILocation
