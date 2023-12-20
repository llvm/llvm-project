; RUN: llc -filetype asm -o - %s | FileCheck %s
; REQUIRES: x86-registered-target

; Ensure that we do not emit any live ranges for the fragment as it is
; bit-sliced which cannot be represented in CodeView.
; CHECK-NOT: .cv_def_range

source_filename = "/tmp/reduced.ll"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc19.37.32825"

define swifttailcc void @"$s12SourceKitLSP0aB6ServerC14workspaceTestsySay08LanguageD8Protocol19WorkspaceSymbolItemOGSgAE0iF7RequestVYaKFTY0_"() !dbg !5 {
entryresume.0:
  br label %.from.63

.from.63:                                         ; preds = %.from.63, %entryresume.0
  %lsr.iv = phi ptr [ %scevgep, %.from.63 ], [ null, %entryresume.0 ]
  %0 = load i64, ptr %lsr.iv, align 8
  %1 = load i64, ptr %lsr.iv, align 8
  call void @llvm.dbg.value(metadata ptr %lsr.iv, metadata !9, metadata !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 1, 1)), !dbg !22
  call void @llvm.dbg.value(metadata ptr %lsr.iv, metadata !9, metadata !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 2, 64)), !dbg !22
  %2 = load volatile ptr, ptr null, align 8, !dbg !30
  %scevgep = getelementptr i8, ptr %lsr.iv, i64 8
  br label %.from.63
}

declare void @llvm.dbg.value(metadata %0, metadata %1, metadata %2) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift version 5.11-dev (LLVM 572a5a4fbafb69e, Swift ebbb9de104d5682)", isOptimized: true, flags: "-private-discriminator _A0745CB010D215CC0C4A68539F8BE5E9", runtimeVersion: 5, emissionKind: FullDebug, imports: !2, sysroot: "S:/Program Files/Swift/Platforms/Windows.platform/Developer/SDKs/Windows.sdk", sdk: "Windows.sdk")
!1 = !DIFile(filename: "S:\\SourceCache\\swift-project\\sourcekit-lsp\\Sources\\SourceKitLSP\\TestDiscovery.swift", directory: "S:\\b\14")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "workspaceTests", linkageName: "$s12SourceKitLSP0aB6ServerC14workspaceTestsySay08LanguageD8Protocol19WorkspaceSymbolItemOGSgAE0iF7RequestVYaKFTY0_", scope: !6, file: !1, line: 35, type: !7, scopeLine: 35, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !8, retainedNodes: !2, thrownTypes: !2)
!6 = !DIModule(scope: null, name: "SourceKitLSP", includePath: "S:\\SourceCache\\swift-project\\sourcekit-lsp\\Sources\\SourceKitLSP")
!7 = !DISubroutineType(types: !2)
!8 = !DISubprogram(name: "workspaceTests", linkageName: "$s12SourceKitLSP0aB6ServerC14workspaceTestsySay08LanguageD8Protocol19WorkspaceSymbolItemOGSgAE0iF7RequestVYaKFTY0_", scope: !6, file: !1, line: 35, type: !7, scopeLine: 35, spFlags: DISPFlagOptimized, thrownTypes: !2)
!9 = !DILocalVariable(name: "$0", arg: 1, scope: !10, file: !1, line: 41, type: !18)
!10 = distinct !DISubprogram(linkageName: "$s12SourceKitLSP0aB6ServerC14workspaceTestsySay08LanguageD8Protocol19WorkspaceSymbolItemOGSgAE0iF7RequestVYaKFSb12IndexStoreDB0J10OccurrenceVXEfU0_", scope: !11, file: !1, line: 41, type: !17, scopeLine: 41, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2, thrownTypes: !13)
!11 = distinct !DISubprogram(name: "workspaceTests", linkageName: "$s12SourceKitLSP0aB6ServerC14workspaceTestsySay08LanguageD8Protocol19WorkspaceSymbolItemOGSgAE0iF7RequestVYaKF", scope: !6, file: !1, line: 35, type: !7, scopeLine: 35, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !12, retainedNodes: !2, thrownTypes: !13)
!12 = !DISubprogram(name: "workspaceTests", linkageName: "$s12SourceKitLSP0aB6ServerC14workspaceTestsySay08LanguageD8Protocol19WorkspaceSymbolItemOGSgAE0iF7RequestVYaKF", scope: !6, file: !1, line: 35, type: !7, scopeLine: 35, spFlags: DISPFlagOptimized, thrownTypes: !13)
!13 = !{!14}
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "Error", scope: !16, file: !15, size: 64, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "$ss5Error_pD")
!15 = !DIFile(filename: "S:\\Program Files\\Swift\\Platforms\\Windows.platform\DEveloper\\SDKs\\Windows.sdk\\usr\\lib\\swift\\windows\\Swift.swiftmodule\\x86_64-unknown-windows-msvc.swiftmodule", directory: "S:\\")
!16 = !DIModule(scope: null, name: "Swift", configMacros: "\22-D_CRT_SECURE_NO_WARNINGS\22 \22-D_MT\22 \22-D_DLL\22", includePath: "S:/Program Files/Swift/Platforms/Windows.platform/Developer/SDKs/Windows.sdk\\usr\\lib\\swift\\windows\\Swift.swiftmodule\\x86_64-unknown-windows-msvc.swiftmodule")
!17 = distinct !DISubroutineType(types: !2)
!18 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !19)
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "SymbolOccurrence", scope: !21, file: !20, size: 960, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "$s12IndexStoreDB16SymbolOccurrenceVD")
!20 = !DIFile(filename: "13\\swift\\IndexStoreDB.swiftmodule", directory: "S:\\b")
!21 = !DIModule(scope: null, name: "IndexStoreDB", configMacros: "\22-D_CRT_SECURE_NO_WARNINGS\22 \22-D_MT\22 \22-D_DLL\22", includePath: "S:\\b\\13\\swift\\IndexStoreDB.swiftmodule")
!22 = !DILocation(line: 41, scope: !10, inlinedAt: !23)
!23 = distinct !DILocation(line: 0, scope: !24, inlinedAt: !28)
!24 = distinct !DISubprogram(name: "filter", linkageName: "$ss14_ArrayProtocolPsE6filterySay7ElementQzGSbAEKXEKFSay12IndexStoreDB16SymbolOccurrenceVG_Tg5", scope: !16, file: !25, type: !26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !27, thrownTypes: !2)
!25 = !DIFile(filename: "<compiler-generated>", directory: "")
!26 = distinct !DISubroutineType(types: !2)
!27 = !DISubprogram(name: "filter", linkageName: "$ss14_ArrayProtocolPsE6filterySay7ElementQzGSbAEKXEKFSay12IndexStoreDB16SymbolOccurrenceVG_Tg5", scope: !16, file: !25, type: !26, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized, thrownTypes: !2)
!28 = distinct !DILocation(line: 41, scope: !29)
!29 = distinct !DILexicalBlock(scope: !5, file: !1, line: 36)
!30 = !DILocation(line: 24, scope: !31, inlinedAt: !34)
!31 = distinct !DILexicalBlock(scope: !32, file: !1, line: 24)
!32 = distinct !DISubprogram(name: "canBeTestDefinition.get", linkageName: "$s12IndexStoreDB16SymbolOccurrenceV12SourceKitLSPE19canBeTestDefinition33_A0745CB010D215CC0C4A68539F8BE5E9LLSbvg", scope: !6, file: !1, line: 23, type: !17, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !33, retainedNodes: !2)
!33 = !DISubprogram(name: "canBeTestDefinition.get", linkageName: "$s12IndexStoreDB16SymbolOccurrenceV12SourceKitLSPE19canBeTestDefinition33_A0745CB010D215CC0C4A68539F8BE5E9LLSbvg", scope: !6, file: !1, line: 23, type: !17, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized)
!34 = distinct !DILocation(line: 41, scope: !10, inlinedAt: !23)
