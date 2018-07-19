; Test to ensure that a module containing both a NoDebug CU and one with
; debug is handled correctly.

; LLVM IR was generated the following way:
; $ cat a.cpp
; void f1();
; __attribute__((always_inline)) void f2() {
;     f1();
; }
; void f3();
; void f4() {
;     f3();
; }
; $ cat b.cpp
; void f2();
; __attribute__((always_inline)) void f3() {
;     f2();
; }
; $ clang++ -flto a.cpp -g -c
; $ clang++ -flto b.cpp -Rpass=inline -c
; $ llvm-link {a,b}.o -o - | opt -O2 - -o ab.bc
; $ llvm-dis ab.bc

; Ensure we can successfully generate assembly, and check that neither
; "b.cpp" nor "f3" strings show up (which would be in the .debug_str
; section if we had generated any lexical scopes and debug for them).
; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s
; CHECK-NOT: .asciz  "b.cpp"
; CHECK-NOT: .asciz  "f3"

; ModuleID = 'debug_and_nodebug_CUs.bc'
source_filename = "debug_and_nodebug_CUs.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_Z2f2v() local_unnamed_addr !dbg !8 {
entry:
  tail call void @_Z2f1v(), !dbg !11
  ret void, !dbg !12
}

declare void @_Z2f1v() local_unnamed_addr

define void @_Z2f4v() local_unnamed_addr !dbg !13 {
entry:
  tail call void @_Z2f1v(), !dbg !14
  ret void, !dbg !19
}

define void @_Z2f3v() local_unnamed_addr !dbg !16 {
entry:
  tail call void @_Z2f1v(), !dbg !20
  ret void, !dbg !22
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 294362) (llvm/trunk 294367)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.cpp", directory: ".")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 5.0.0 (trunk 294362) (llvm/trunk 294367)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!4 = !DIFile(filename: "b.cpp", directory: ".")
!5 = !{!"clang version 5.0.0 (trunk 294362) (llvm/trunk 294367)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 3, column: 3, scope: !8)
!12 = !DILocation(line: 4, column: 1, scope: !8)
!13 = distinct !DISubprogram(name: "f4", linkageName: "_Z2f4v", scope: !1, file: !1, line: 6, type: !9, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!14 = !DILocation(line: 3, column: 3, scope: !8, inlinedAt: !15)
!15 = distinct !DILocation(line: 3, column: 3, scope: !16, inlinedAt: !18)
!16 = distinct !DISubprogram(name: "f3", scope: !4, file: !4, line: 2, type: !17, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !3, variables: !2)
!17 = !DISubroutineType(types: !2)
!18 = distinct !DILocation(line: 7, column: 3, scope: !13)
!19 = !DILocation(line: 8, column: 1, scope: !13)
!20 = !DILocation(line: 3, column: 3, scope: !8, inlinedAt: !21)
!21 = distinct !DILocation(line: 3, column: 3, scope: !16)
!22 = !DILocation(line: 4, column: 1, scope: !16)
