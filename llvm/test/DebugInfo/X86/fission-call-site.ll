; Check that call site entries are emitted correctly when using split-dwarf
; together with some form of LTO.

; Original C source:
;
; // cu1.c
; extern void callee(void);
; void caller(void) {
;   callee();
; }
;
; // cu2.c
; __attribute__((optnone)) void callee(void) {}
;
; Steps to reproduce:
;
; (The -O1 here is needed to trigger call site entry emission, as these tags are
; generally not emitted at -O0.)
;
; clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=split -O1 ~/tmp/cu1.c -S -emit-llvm -o ~/tmp/cu1.ll
; clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=split -O1 ~/tmp/cu2.c -S -emit-llvm -o ~/tmp/cu2.ll
; llvm-link -o ~/tmp/cu-merged.bc ~/tmp/cu1.ll ~/tmp/cu2.ll
; llc -split-dwarf-file=foo.dwo -O0 -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - ~/tmp/cu-merged.bc

; RUN: llc -split-dwarf-file=foo.dwo -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: DW_TAG_GNU_call_site
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "callee"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @caller() local_unnamed_addr !dbg !14 {
entry:
  call void @callee(), !dbg !15
  ret void, !dbg !16
}

define dso_local void @callee() local_unnamed_addr noinline optnone !dbg !17 {
entry:
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!0, !8}
!llvm.ident = !{!10, !10}
!llvm.module.flags = !{!11, !12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (git@github.com:llvm/llvm-project.git 170f4b972e7bcf1f2af98bdd7145954efd16e038)", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "cu1.dwo", emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "/Users/vsk/tmp/cu1.c", directory: "/Users/vsk/src/builds/llvm-project-master-RA")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "callee", scope: !5, file: !5, line: 1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DIFile(filename: "tmp/cu1.c", directory: "/Users/vsk")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !9, producer: "clang version 11.0.0 (git@github.com:llvm/llvm-project.git 170f4b972e7bcf1f2af98bdd7145954efd16e038)", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "cu2.dwo", emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: GNU)
!9 = !DIFile(filename: "/Users/vsk/tmp/cu2.c", directory: "/Users/vsk/src/builds/llvm-project-master-RA")
!10 = !{!"clang version 11.0.0 (git@github.com:llvm/llvm-project.git 170f4b972e7bcf1f2af98bdd7145954efd16e038)"}
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = distinct !DISubprogram(name: "caller", scope: !5, file: !5, line: 2, type: !6, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 3, column: 3, scope: !14)
!16 = !DILocation(line: 4, column: 1, scope: !14)
!17 = distinct !DISubprogram(name: "callee", scope: !18, file: !18, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !2)
!18 = !DIFile(filename: "tmp/cu2.c", directory: "/Users/vsk")
!19 = !DILocation(line: 1, column: 45, scope: !17)
