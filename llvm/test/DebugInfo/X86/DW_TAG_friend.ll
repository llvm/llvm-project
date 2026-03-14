; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Check that the friend tag is there and is followed by a DW_AT_friend that has a reference back.

; CHECK: [[BACK:0x[0-9a-f]*]]:   DW_TAG_class_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]       ( .debug_str[{{.*}}] = "A")
; CHECK: DW_TAG_friend
; CHECK-NEXT: DW_AT_friend [DW_FORM_ref4]   (cu + 0x{{[0-9a-f]*}} => {[[BACK]]})

source_filename = "test/DebugInfo/X86/DW_TAG_friend.ll"

%class.A = type { i32 }
%class.B = type { i32 }

@a = global %class.A zeroinitializer, align 4, !dbg !0
@b = global %class.B zeroinitializer, align 4, !dbg !11

!llvm.dbg.cu = !{!21}
!llvm.module.flags = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 10, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")
!3 = !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !2, line: 1, size: 32, align: 32, elements: !4)
!4 = !{!5, !7}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !2, line: 2, baseType: !6, size: 32, align: 32, flags: DIFlagPrivate)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DISubprogram(name: "A", scope: !3, file: !2, line: 1, type: !8, isLocal: false, isDefinition: false, scopeLine: 1, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 11, type: !13, isLocal: false, isDefinition: true)
!13 = !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !2, line: 5, size: 32, align: 32, elements: !14)
!14 = !{!15, !16, !20}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !2, line: 7, baseType: !6, size: 32, align: 32, flags: DIFlagPrivate)
!16 = !DISubprogram(name: "B", scope: !13, file: !2, line: 5, type: !17, isLocal: false, isDefinition: false, scopeLine: 5, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, align: 64, flags: DIFlagArtificial)
!20 = !DIDerivedType(tag: DW_TAG_friend, file: !2, baseType: !3)
!21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.1 (trunk 153413) (llvm/trunk 153428)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !22, retainedTypes: !22, globals: !23, imports: !22)
!22 = !{}
!23 = !{!0, !11}
!24 = !{i32 1, !"Debug Info Version", i32 3}

