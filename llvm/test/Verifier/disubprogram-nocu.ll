; Reject if DISubprogram does not belong to a DICompileUnit.
; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s

; CHECK:  assembly parsed, but does not verify as correct!
; CHECK: subprogram not found in any compile unit

@_ZZNK4llvm6object15MachOObjectFile21getRelocationTypeNameENS0_11DataRefImplERNS_15SmallVectorImplIcEEE5Table = external unnamed_addr constant [6 x i8*], align 16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: 0, globals: !2, imports: !9)
!1 = !DIFile(filename: "../lib/Object/MachOObjectFile.cpp", directory: "/home/davide/work/llvm/build-lto-debug")
!2 = !{!3, !8}
!3 = !DIGlobalVariable(name: "Table", scope: !4, isLocal: false, isDefinition: true, variable: [6 x i8*]* @_ZZNK4llvm6object15MachOObjectFile21getRelocationTypeNameENS0_11DataRefImplERNS_15SmallVectorImplIcEEE5Table)
!4 = distinct !DILexicalBlock(scope: !5, line: 722, column: 23)
!5 = distinct !DILexicalBlock(scope: !6, line: 721, column: 17)
!6 = distinct !DISubprogram(name: "getRelocationTypeName", scope: null, isLocal: false, isDefinition: true, isOptimized: false, variables: !7)
!7 = !{}
!8 = !DIGlobalVariable(name: "IsLittleEndianHost", scope: null, isLocal: false, isDefinition: true, variable: i1 true)
!9 = !{!10, !12}
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, line: 121)
!11 = !DINamespace(name: "std", scope: null, line: 1967)
!12 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0, line: 32)
!13 = !{i32 2, !"Debug Info Version", i32 3}
