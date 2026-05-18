; Test direct basename matching for orphan functions where multiple callee anchors may be
; matched to one same profile during stale profile matching. In this test case, both of the
; `mid` functions will be matched to the same `_Z3midi` function in the profile during stale
; profile matching. This ends up causing an assertation error because only one profile
; function is supposed to be matched to an IR function.
;
; IR Function:
;   foo: top ; bar ; top(2)
;         |_mid       |_ mid(2)
;                          |_ sub
; 
; Profile Function:
;   foo: bar ; top ; mid
;                     |_ sub
; 
; Stale profile match order:
;   foo:  top<ir>   ; bar<ir>   ; top(2)<ir>
;            |           |           |
;         top<prof> ; bar<prof> ; top<prof>
;
;   top(2)<ir>:  mid(2)<ir>
;                  |
;                mid<prof>
;
;   top<ir>:  mid<ir>
;               |
;             mid<prof>    => (Assertation error)


; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-stale-profile-orphan-conflict.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl 2>&1 | FileCheck %s

; CHECK: Function _Z3fool is not in profile or profile symbol list.
; CHECK: Function _Z3topl is not in profile or profile symbol list.
; CHECK: Function _Z3barl is not in profile or profile symbol list.
; CHECK: Function _Z3topll is not in profile or profile symbol list.
; CHECK: Function _Z3midl is not in profile or profile symbol list.
; CHECK: Function _Z3midll is not in profile or profile symbol list.
; CHECK: Direct basename match: _Z3barl (IR) -> _Z3bari (Profile) [basename: bar]
; CHECK: Direct basename match: _Z3fool (IR) -> _Z3fooi (Profile) [basename: foo]
; CHECK: Direct basename matching found 2 matches
; CHECK: Run stale profile matching for _Z3fool
; CHECK: The functions _Z3topl(IR) and _Z3topi(Profile) share the same base name: top.
; CHECK: Function:_Z3topl matches profile:_Z3topi
; CHECK: The functions _Z3barl(IR) and _Z3bari(Profile) share the same base name: bar.
; CHECK: Function:_Z3barl matches profile:_Z3bari
; CHECK: The functions _Z3topll(IR) and _Z3topi(Profile) share the same base name: top.
; CHECK: Function:_Z3topll matches profile:_Z3topi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

define ptr @_Z3fool(i64 %x) #0 {
entry:
  br i1 false, label %if.then, label %if.end63

if.then:                                          ; preds = %entry
  %call45 = call ptr @_Z3topl(i64 0), !dbg !5
  ret ptr %call45

if.end63:                                         ; preds = %entry
  call void @_Z3barl(i64 0), !dbg !15
  %call103 = call ptr @_Z3topll(i64 0, i64 1), !dbg !19
  ret ptr null
}

define ptr @_Z3topl(i64 %x) #0 {
entry:
  %call = call ptr @_Z3midl(i64 0), !dbg !21
  ret ptr %call
}

define void @_Z3barl(i64 %x) {
entry:
  ret void
}

define ptr @_Z3topll(i64 %x, i64 %y) #0 {
entry:
  %call = call ptr @_Z3midll(i64 0, i64 1), !dbg !25
  ret ptr %call
}

define ptr @_Z3midl(i64 %x) #0 {
entry:
  ret ptr null
}

define ptr @_Z3midll(i64 %x, i64 %y) #0 {
entry:
  %call18 = call ptr @_Z3subi(i64 0), !dbg !29
  ret ptr %call18
}

define ptr @_Z3subi(i64 %x) {
entry:
  ret ptr null
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.pseudo_probe_desc = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 23.0.0git (https://github.com/llvm/llvm-project.git 4e38fa748a8de3a7bc2d8c27bb43f53ea78af395)", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "out.ll", emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2, splitDebugInlining: false, debugInfoForProfiling: true)
!1 = !DIFile(filename: "tree.cpp", directory: ".", checksumkind: CSK_MD5, checksum: "dbfcb2083a71f1367256c8fcc6263915")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i64 -4689275162925682552, i64 2251974424712944, !"_Z3midll"}
!5 = !DILocation(line: 2378, scope: !6, atomGroup: 28279, atomRank: 2)
!6 = !DILexicalBlockFile(scope: !8, file: !7, discriminator: 455082215)
!7 = !DIFile(filename: "format.h", directory: ".", checksumkind: CSK_MD5, checksum: "24094ed6f9970cebe9f2fedff5aac8b9")
!8 = distinct !DILexicalBlock(scope: !9, file: !7, line: 2353)
!9 = distinct !DILexicalBlock(scope: !10, file: !7, line: 2353)
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fool", scope: !11, file: !7, line: 2331, type: !14, scopeLine: 2333, flags: DIFlagPrototyped | DIFlagAllCallsDescribed | DIFlagNameIsSimplified, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2, keyInstructions: true)
!11 = !DINamespace(name: "c", scope: !12)
!12 = !DINamespace(name: "b", scope: !13, exportSymbols: true)
!13 = !DINamespace(name: "a", scope: null)
!14 = distinct !DISubroutineType(types: !2)
!15 = !DILocation(line: 2394, scope: !16)
!16 = !DILexicalBlockFile(scope: !17, file: !7, discriminator: 455147919)
!17 = distinct !DILexicalBlock(scope: !18, file: !7, line: 2383)
!18 = distinct !DILexicalBlock(scope: !10, file: !7, line: 2383)
!19 = !DILocation(line: 2396, scope: !20, atomGroup: 28280, atomRank: 2)
!20 = !DILexicalBlockFile(scope: !17, file: !7, discriminator: 455082407)
!21 = !DILocation(line: 1652, scope: !22, atomGroup: 28281, atomRank: 2)
!22 = !DILexicalBlockFile(scope: !23, file: !7, discriminator: 455082007)
!23 = distinct !DISubprogram(name: "top", linkageName: "_Z3topl", scope: !11, file: !7, line: 1650, type: !24, scopeLine: 1651, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2, keyInstructions: true)
!24 = distinct !DISubroutineType(types: !2)
!25 = !DILocation(line: 1652, scope: !26, atomGroup: 28282, atomRank: 2)
!26 = !DILexicalBlockFile(scope: !27, file: !7, discriminator: 455082007)
!27 = distinct !DISubprogram(name: "top", linkageName: "_Z3topll", scope: !11, file: !7, line: 1650, type: !28, scopeLine: 1651, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2, keyInstructions: true)
!28 = distinct !DISubroutineType(types: !2)
!29 = !DILocation(line: 1643, scope: !30)
!30 = !DILexicalBlockFile(scope: !31, file: !7, discriminator: 455082087)
!31 = distinct !DISubprogram(name: "mid", linkageName: "_Z3midll", scope: !11, file: !7, line: 1629, type: !32, scopeLine: 1630, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2, keyInstructions: true)
!32 = distinct !DISubroutineType(types: !2)
