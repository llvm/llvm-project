; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/csspgo-import-list-preinliner.prof -S -profile-summary-cutoff-hot=100000 -sample-profile-use-preinliner=0 | FileCheck %s --check-prefix=DISABLE-PREINLINE
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/csspgo-import-list-preinliner.prof -S -profile-summary-cutoff-hot=100000 | FileCheck %s

; The GUID of bar is -2012135647395072713

; DISABLE-PREINLINE-NOT: -2012135647395072713
; CHECK: [[#]] = !{!"function_entry_count", i64 1, i64 -2012135647395072713}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  call void @llvm.pseudoprobe(i64 0, i64 0, i32 0, i64 0)
  %call2 = call i32 @bar(), !dbg !9
  br label %for.cond
}

declare i32 @bar()

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}
!llvm.pseudo_probe_desc = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/", checksumkind: CSK_MD5, checksum: "1bff37d8b3f7858b0bc29ab4efdf9422")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(name: "x", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true)
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i64 -2624081020897602054, i64 563108639284859, !"main"}
!9 = !DILocation(line: 11, column: 10, scope: !10)
!10 = !DILexicalBlockFile(scope: !11, file: !1, discriminator: 186646615)
!11 = distinct !DILexicalBlock(scope: !12, file: !1, line: 8, column: 40)
!12 = distinct !DILexicalBlock(scope: !13, file: !1, line: 8, column: 3)
!13 = distinct !DILexicalBlock(scope: !14, file: !1, line: 8, column: 3)
!14 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !15, scopeLine: 7, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!15 = distinct !DISubroutineType(types: !16)
!16 = !{}
