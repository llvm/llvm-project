; REQUIRES: asserts && x86-registered-target
; RUN: opt < %s -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline  -sample-profile-file=%S/Inputs/pseudo-probe-callee-profile-mismatch.prof --salvage-stale-profile --salvage-unused-profile=false -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl  -pass-remarks=inline 2>&1 | FileCheck %s

; There is no profile-checksum-mismatch attr, even the checksum is mismatched in the pseudo_probe_desc, it doesn't run the matching.
; CHECK-NOT: Run stale profile matching for main

; CHECK: Run stale profile matching for bar
; CHECK: Callsite with callee:baz is matched from 4 to 2
; CHECK: 'baz' inlined into 'main' to match profiling context with (cost=always): preinliner at callsite bar:3:8.4 @ main:3:10.7

; CHECK: Probe descriptor missing for Function bar
; CHECK: Profile is invalid due to CFG mismatch for Function bar


define available_externally i32 @main() #0 {
  %1 = call i32 @bar(), !dbg !13
  ret i32 0
}

define available_externally i32 @bar() #1 !dbg !21 {
  %1 = call i32 @baz(), !dbg !23
  ret i32 0
}

define available_externally i32 @baz() #0 !dbg !25 {
  ret i32 0
}

attributes #0 = { "use-sample-profile" }
attributes #1 = { "profile-checksum-mismatch" "use-sample-profile" }

!llvm.dbg.cu = !{!0, !7, !9}
!llvm.module.flags = !{!11}
!llvm.pseudo_probe_desc = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/test", checksumkind: CSK_MD5, checksum: "7220f1a2d70ff869f1a6ab7958e3c393")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(name: "x", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true)
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !8, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!8 = !DIFile(filename: "test1.v1.c", directory: "/home/test", checksumkind: CSK_MD5, checksum: "76696bd6bfe16a9f227fe03cfdb6a82c")
!9 = distinct !DICompileUnit(language: DW_LANG_C11, file: !10, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!10 = !DIFile(filename: "test2.c", directory: "/home/test", checksumkind: CSK_MD5, checksum: "553093afc026f9c73562eb3b0c5b7532")
!11 = !{i32 2, !"Debug Info Version", i32 3}
; Make a checksum mismatch in the pseudo_probe_desc
!12 = !{i64 -2624081020897602054, i64 123456, !"main"}
!13 = !DILocation(line: 8, column: 10, scope: !14)
!14 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 186646591)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 7, column: 40)
!16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 7, column: 3)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 7, column: 3)
!18 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !19, scopeLine: 6, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!19 = distinct !DISubroutineType(types: !20)
!20 = !{}
!21 = distinct !DISubprogram(name: "bar", scope: !8, file: !8, line: 3, type: !22, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !20)
!22 = !DISubroutineType(types: !20)
!23 = !DILocation(line: 6, column: 8, scope: !24)
!24 = !DILexicalBlockFile(scope: !21, file: !8, discriminator: 186646567)
!25 = distinct !DISubprogram(name: "baz", scope: !10, file: !10, line: 1, type: !22, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !20)
