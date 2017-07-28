; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-objdump -h - | FileCheck %s
; REQUIRES: default_triple

; CHECK-NOT: .debug_

!llvm.dbg.cu = !{!0, !5}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: !6)
!6 = !{!7}
!7 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !8, entity: !8, file: !1, line: 3)
!8 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !5, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DINamespace(name: "ns", scope: null)
