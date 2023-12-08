; Test that llvm-reduce can remove uninteresting DI metadata from an IR file.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=di-metadata --test=FileCheck --test-arg=--check-prefix=CHECK-INTERESTINGNESS --test-arg=%s --test-arg=--input-file %s -o %t
; RUN: FileCheck <%t %s

@global = global i32 0

define void @main() !dbg !5 {
   ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!4}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "test.c", directory: "test")
!2 = !{!10}
; CHECK: [[EMPTY:![0-9]+]] = !{}
!4 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !9)
!5 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 498, type: !6, scopeLine: 0, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
; CHECK: !DISubroutineType(types: [[EMPTY]])
!7 = !{!13}
!8 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !9)
!9 = !{}
!10 = !DILocalVariable(name: "one", scope: !14, file: !1, line: 0, type: !15)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3)
!15 = !DICompositeType(tag: DW_TAG_structure_type, name: "test", file: !1, size: 64, align: 32, flags: DIFlagPublic, elements: !16)
!16 = !{!17, !18}
; CHECK: elements: [[EL:![0-9]+]])
; CHECK: [[EL]] = !{!{{[0-9]+}}}
; CHECK-INTERESTINGNESS: interesting
!17 = !DIDerivedType(tag: DW_TAG_member, name: "interesting", scope: !14, file: !1, baseType: !13, size: 32, align: 32, flags: DIFlagPublic)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "uninteresting", scope: !14, file: !1, baseType: !13, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
