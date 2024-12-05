; RUN: opt %s -o - -S | FileCheck %s

; Paired with clang/test/CodeGenCXX/debug-info-local-types.cpp, this round-trips
; debug-info metadata for types that are ODR-uniqued, to ensure that the type
; hierachy does not change. See the enableDebugTypeODRUniquing feature: types
; that have the "identifier" field set will be unique'd based on their name,
; even if the "distinct" flag is set. Clang doesn't enable that itself, but opt
; does, therefore we pass the metadata through opt to check it doesn't change
; the type hiearchy.
;
; The check-lines below are not strictly in order of hierachy, so here's a
; diagram of what's desired:
;
;                  DIFile
;                    |
;          Decl-DISubprogram "foo"
;          /                      \
;         /                        \
; Def-DISubprogram "foo"    DICompositeType "bar"
;                                   |
;                                   |
;                          Decl-DISubprogram "get_a"
;                         /         |
;                        /          |
; Def-DISubprogram "get_a"    DICompositeType "baz"
;                                   |
;                                   |
;                        {Def,Decl}-DISubprogram "get_b"
;
; The declaration DISubprograms are unique'd, and the DICompositeTypes should
; be in those scopes rather than the definition DISubprograms.

; CHECK: ![[FILENUM:[0-9]+]] = !DIFile(filename: "{{.*}}debug-info-local-types.cpp",

; CHECK: ![[BARSTRUCT:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "bar", scope: ![[FOOFUNC:[0-9]+]], file: ![[FILENUM]],
; CHECK-SAME: identifier: "_ZTSZ3foovE3bar")

; CHECK: ![[FOOFUNC]] = !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: ![[FILENUM]], file: ![[FILENUM]],
;; Test to ensure that this is _not_ a definition, therefore a decl.
; CHECK-SAME: spFlags: 0)

; CHECK: ![[GETADECL:[0-9]+]] = !DISubprogram(name: "get_a", scope: ![[BARSTRUCT]], file: ![[FILENUM]],
;; Test to ensure that this is _not_ a definition, therefore a decl.
; CHECK-SAME: spFlags: 0)

; CHECK: ![[BAZSTRUCT:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "baz", scope: ![[GETADECL]], file: ![[FILENUM]],
; CHECK-SAME: identifier: "_ZTSZZ3foovEN3bar5get_aEvE3baz")
; CHECK: distinct !DISubprogram(name: "get_b",
; CHECK-SAME: scope: ![[BAZSTRUCT]], file: ![[FILENUM]],

%class.bar = type { i32 }
%class.baz = type { i32 }

$_Z3foov = comdat any

$_ZZ3foovEN3bar5get_aEv = comdat any

$_ZZZ3foovEN3bar5get_aEvEN3baz5get_bEv = comdat any

$_ZZ3foovE3baz = comdat any

$_ZZZ3foovEN3bar5get_aEvE5xyzzy = comdat any

@_ZZ3foovE3baz = linkonce_odr global %class.bar zeroinitializer, comdat, align 4, !dbg !0
@_ZZZ3foovEN3bar5get_aEvE5xyzzy = linkonce_odr global %class.baz zeroinitializer, comdat, align 4, !dbg !10

define dso_local noundef i32 @_Z1av() !dbg !32 {
entry:
  unreachable
}

define linkonce_odr noundef i32 @_Z3foov() comdat !dbg !2 {
entry:
  unreachable
}

define linkonce_odr noundef i32 @_ZZ3foovEN3bar5get_aEv(ptr noundef nonnull align 4 dereferenceable(4) %this) comdat align 2 !dbg !12 {
entry:
  unreachable
}

define linkonce_odr noundef i32 @_ZZZ3foovEN3bar5get_aEvEN3baz5get_bEv(ptr noundef nonnull align 4 dereferenceable(4) %this) comdat align 2 !dbg !33 {
entry:
  unreachable
}

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!29, !30}
!llvm.ident = !{!31}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "baz", scope: !2, file: !3, line: 71, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 51, type: !4, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, declaration: !14)
!3 = !DIFile(filename: "debug-info-local-types.cpp", directory: ".")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !9, splitDebugInlining: false, nameTableKind: None)
!8 = !DIFile(filename: "<stdin>", directory: ".")
!9 = !{!0, !10}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "xyzzy", scope: !12, file: !3, line: 66, type: !22, isLocal: false, isDefinition: true)
!12 = distinct !DISubprogram(name: "get_a", linkageName: "_ZZ3foovEN3bar5get_aEv", scope: !13, file: !3, line: 56, type: !18, scopeLine: 56, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, declaration: !17, retainedNodes: !21)
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "bar", scope: !14, file: !3, line: 52, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !15, identifier: "_ZTSZ3foovE3bar")
!14 = !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 51, type: !4, scopeLine: 51, flags: DIFlagPrototyped, spFlags: 0)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !3, line: 54, baseType: !6, size: 32)
!17 = !DISubprogram(name: "get_a", scope: !13, file: !3, line: 56, type: !18, scopeLine: 56, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!18 = !DISubroutineType(types: !19)
!19 = !{!6, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !{}
!22 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "baz", scope: !17, file: !3, line: 57, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !23, identifier: "_ZTSZZ3foovEN3bar5get_aEvE3baz")
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !22, file: !3, line: 59, baseType: !6, size: 32)
!25 = !DISubprogram(name: "get_b", scope: !22, file: !3, line: 61, type: !26, scopeLine: 61, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!26 = !DISubroutineType(types: !27)
!27 = !{!6, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 1, !"wchar_size", i32 4}
!31 = !{!"clang"}
!32 = distinct !DISubprogram(name: "a", linkageName: "_Z1av", scope: !3, file: !3, line: 75, type: !4, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7)
!33 = distinct !DISubprogram(name: "get_b", linkageName: "_ZZZ3foovEN3bar5get_aEvEN3baz5get_bEv", scope: !22, file: !3, line: 61, type: !26, scopeLine: 61, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, declaration: !25, retainedNodes: !21)
