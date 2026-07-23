; REQUIRES: x86_64-linux
; RUN: llvm-profdata merge %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdata
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pass-remarks-analysis=memprof -S 2>&1 | FileCheck %s

; CHECK: remark: memprof.cc:5:10: frame: [[FOO:[0-9]+]] _Z3foov:1:10
; CHECK: remark: memprof.cc:5:10: inline call stack: [[FOO]]
; CHECK: remark: memprof.cc:9:12: frame: [[BAR:[0-9]+]] _Z3barv:2:12
; CHECK: remark: memprof.cc:9:12: frame: [[BAZ:[0-9]+]] _Z3bazv:3:13
; CHECK: remark: memprof.cc:9:12: inline call stack: [[BAR]],[[BAZ]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @_Z3foov() {
entry:
  %call = call ptr null(i64 0), !dbg !3
  ret ptr %call
}

define ptr @_Z3barv() {
entry:
  %call = call ptr @_Z3foov(), !dbg !7
  ret ptr %call
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 6cbe6284d1f0a088b5c6482ae27b738f03d82fe7)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof.cc", directory: "/", checksumkind: CSK_MD5, checksum: "e8c40ebe4b21776b4d60e9632cbc13c2")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocation(line: 5, column: 10, scope: !4)
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !5, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 9, column: 12, scope: !8, inlinedAt: !9)
!8 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 7, type: !5, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !6)
!9 = !DILocation(line: 12, column: 13, scope: !10)
!10 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 9, type: !5, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !6)
