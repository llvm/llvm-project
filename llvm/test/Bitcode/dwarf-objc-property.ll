; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: !DIObjCProperty(name: "autoSynthProp", file: !3, line: 5, attributes: 2316, type: !8)
; CHECK: !DIObjCProperty(name: "synthProp", file: !3, line: 6, attributes: 2316, type: !8)
; CHECK: !DIObjCProperty(name: "customGetterProp", file: !3, line: 7, getter: "customGetter", attributes: 2318, type: !8)
; CHECK: !DIObjCProperty(name: "customSetterProp", file: !3, line: 8, setter: "customSetter:", attributes: 2444, type: !8)
; CHECK: !DIObjCProperty(name: "customAccessorsProp", file: !3, line: 9, setter: "customSetter:", getter: "customGetter", attributes: 2446, type: !8)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !3, producer: "hand written", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, retainedTypes: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: Apple)
!3 = !DIFile(filename: "main.m", directory: "/tmp")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", scope: !3, file: !3, line: 1, size: 128, flags: DIFlagObjcClassComplete, elements: !6, runtimeLang: DW_LANG_ObjC)
!6 = !{!7, !9, !10, !11, !12, !13, !14, !15, !16, !17, !24, !27, !28, !29, !30, !31, !32}
!7 = !DIObjCProperty(name: "autoSynthProp", file: !3, line: 5, attributes: 2316, type: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIObjCProperty(name: "synthProp", file: !3, line: 6, attributes: 2316, type: !8)
!10 = !DIObjCProperty(name: "customGetterProp", file: !3, line: 7, getter: "customGetter", attributes: 2318, type: !8)
!11 = !DIObjCProperty(name: "customSetterProp", file: !3, line: 8, setter: "customSetter:", attributes: 2444, type: !8)
!12 = !DIObjCProperty(name: "customAccessorsProp", file: !3, line: 9, setter: "customSetter:", getter: "customGetter", attributes: 2446, type: !8)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "someBackingIvar", scope: !3, file: !3, line: 2, baseType: !8, size: 32, flags: DIFlagProtected, extraData: !9)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "_autoSynthProp", scope: !3, file: !3, line: 5, baseType: !8, size: 32, flags: DIFlagPrivate, extraData: !7)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "_customGetterProp", scope: !3, file: !3, line: 7, baseType: !8, size: 32, flags: DIFlagPrivate, extraData: !10)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "_customSetterProp", scope: !3, file: !3, line: 8, baseType: !8, size: 32, flags: DIFlagPrivate, extraData: !11)
!17 = !DISubprogram(name: "-[Foo customGetter]", scope: !5, file: !3, line: 19, type: !18, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!18 = !DISubroutineType(types: !19)
!19 = !{!8, !20, !21}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", file: !3, baseType: !22, flags: DIFlagArtificial)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_selector", file: !3, flags: DIFlagFwdDecl)
!24 = !DISubprogram(name: "-[Foo customSetter:]", scope: !5, file: !3, line: 23, type: !25, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !20, !21, !8}
!27 = !DISubprogram(name: "-[Foo synthProp]", scope: !5, file: !3, line: 17, type: !18, scopeLine: 17, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!28 = !DISubprogram(name: "-[Foo setSynthProp:]", scope: !5, file: !3, line: 17, type: !25, scopeLine: 17, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!29 = !DISubprogram(name: "-[Foo autoSynthProp]", scope: !5, file: !3, line: 5, type: !18, scopeLine: 5, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!30 = !DISubprogram(name: "-[Foo setAutoSynthProp:]", scope: !5, file: !3, line: 5, type: !25, scopeLine: 5, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!31 = !DISubprogram(name: "-[Foo setCustomGetterProp:]", scope: !5, file: !3, line: 7, type: !25, scopeLine: 7, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!32 = !DISubprogram(name: "-[Foo customSetterProp]", scope: !5, file: !3, line: 8, type: !18, scopeLine: 8, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)

