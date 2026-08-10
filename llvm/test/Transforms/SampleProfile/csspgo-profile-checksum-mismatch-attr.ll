; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/pseudo-probe-callee-profile-mismatch.prof -pass-remarks=inline  -S -o %t 2>&1 | FileCheck %s --check-prefix=INLINE
; RUN: FileCheck %s < %t
; RUN: FileCheck %s < %t --check-prefix=MERGE


; Make sure bar is inlined into main for attr merging verification.
; INLINE: 'bar' inlined into 'main'

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @baz() #0 {
entry:
  ret i32 0
}

define i32 @bar() #0 !dbg !11 {
; CHECK: define {{.*}} @bar() {{.*}} #[[#BAR_ATTR:]] !
entry:
  %call = call i32 @baz()
  ret i32 0
}

define i32 @main() #0 {
; MERGE: define {{.*}} @main() {{.*}} #[[#MAIN_ATTR:]] !
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call = call i32 @bar(), !dbg !14
  br label %for.cond
}

; CHECK: attributes #[[#BAR_ATTR]] = {{{.*}} "profile-checksum-mismatch" {{.*}}}

; Verify the attribute is not merged into the caller.
; MERGE-NOT: attributes #[[#MAIN_ATTR]] = {{{.*}} "profile-checksum-mismatch" {{.*}}}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}
!llvm.pseudo_probe_desc = !{!8, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home", checksumkind: CSK_MD5, checksum: "0df0c950a93a603a7d13f0a9d4623642")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(name: "x", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true)
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i64 7546896869197086323, i64 4294967295, !"baz"}
!9 = !{i64 -2012135647395072713, i64 281530612780802, !"bar"}
!10 = !{i64 -2624081020897602054, i64 281582081721716, !"main"}
!11 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!12 = distinct !DISubroutineType(types: !13)
!13 = !{}
!14 = !DILocation(line: 15, column: 10, scope: !15)
!15 = !DILexicalBlockFile(scope: !16, file: !1, discriminator: 186646591)
!16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 14, column: 40)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 14, column: 3)
!18 = distinct !DILexicalBlock(scope: !19, file: !1, line: 14, column: 3)
!19 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !20, scopeLine: 13, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!20 = !DISubroutineType(types: !13)
