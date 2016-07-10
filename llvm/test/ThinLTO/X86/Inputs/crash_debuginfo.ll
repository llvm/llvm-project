; ModuleID = 'src-reduced.bc'
source_filename = "src.bc"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

%"class.boost::optional.98.377" = type { %"class.boost::optional_detail::optional_base.97.376" }
%"class.boost::optional_detail::optional_base.97.376" = type { i8, [7 x i8], %"class.boost::optional_detail::aligned_storage.96.375" }
%"class.boost::optional_detail::aligned_storage.96.375" = type { %"union.boost::optional_detail::aligned_storage<TDebugOption>::dummy_u.95.374" }
%"union.boost::optional_detail::aligned_storage<TDebugOption>::dummy_u.95.374" = type { [16 x i8] }
%struct.__sFILE.3.431 = type { i8*, i32, i32, i16, i16, %struct.__sbuf.1.429, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf.1.429, %struct.__sFILEX.2.430*, i32, [3 x i8], [1 x i8], %struct.__sbuf.1.429, i32, i64 }
%struct.__sFILEX.2.430 = type opaque
%struct.__sbuf.1.429 = type { i8*, i32 }

define void @_ZN13ADebugOptions9GetOptionEPKc(%"class.boost::optional.98.377"* noalias nocapture sret %arg) {
  %tmp = getelementptr inbounds %"class.boost::optional.98.377", %"class.boost::optional.98.377"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i64 0, !dbg !22
  unreachable
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!57}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "Apple LLVM version 8.0.0 (clang-800.0.24.1)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "/Stk/Ableton/img/Live/master/Projects/Base/Unity/Base_Unity_Src_cpp-2.cpp", directory: "/Stk/Ableton/img/Live/master")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "sTemporaryTraceStream", linkageName: "_ZN12_GLOBAL__N_1L21sTemporaryTraceStreamE", scope: !5, file: !6, line: 52, type: !7, isLocal: true, isDefinition: true, variable: %struct.__sFILE.3.431** undef)
!5 = !DINamespace(scope: null, file: !6, line: 42)
!6 = !DIFile(filename: "Projects/Base/Src/BaseDebug.cpp", directory: "/Stk/Ableton/img/Live/master")
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !9, line: 153, baseType: !10)
!9 = !DIFile(filename: "/Applications/Xcode-8-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk/usr/include/stdio.h", directory: "/Stk/Ableton/img/Live/master")
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__sFILE", file: !9, line: 122, size: 1216, align: 64, elements: !2, identifier: "_ZTS7__sFILE")
!11 = !DILocation(line: 686, column: 14, scope: !12, inlinedAt: !17)
!12 = distinct !DISubprogram(name: "QueryTop<(anonymous namespace)::TOptionGetter>", linkageName: "_ZNK12_GLOBAL__N_118TDebugOptionsStack8QueryTopINS_13TOptionGetterEEENSt3__19result_ofIFT_RKNS_13TDebugOptionsEEE4typeES5_", scope: !14, file: !13, line: 676, type: !15, isLocal: true, isDefinition: true, scopeLine: 677, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !2, declaration: !16, variables: !2)
!13 = !DIFile(filename: "Projects/Base/Src/BaseDebugOptions.cpp", directory: "/Stk/Ableton/img/Live/master")
!14 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "TDebugOptionsStack", scope: !5, file: !13, line: 667, size: 448, align: 64, elements: !2)
!15 = !DISubroutineType(types: !2)
!16 = !DISubprogram(name: "QueryTop<(anonymous namespace)::TOptionGetter>", linkageName: "_ZNK12_GLOBAL__N_118TDebugOptionsStack8QueryTopINS_13TOptionGetterEEENSt3__19result_ofIFT_RKNS_13TDebugOptionsEEE4typeES5_", scope: !14, file: !13, line: 676, type: !15, isLocal: false, isDefinition: false, scopeLine: 676, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true, templateParams: !2)
!17 = distinct !DILocation(line: 1840, column: 10, scope: !18)
!18 = distinct !DISubprogram(name: "GetOption", linkageName: "_ZN13ADebugOptions9GetOptionEPKc", scope: !19, file: !13, line: 1826, type: !15, isLocal: false, isDefinition: true, scopeLine: 1827, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !21, variables: !2)
!19 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "ADebugOptions", file: !20, line: 111, size: 8, align: 8, elements: !2, identifier: "_ZTS13ADebugOptions")
!20 = !DIFile(filename: "Projects/Base/Exp/BaseDebugOptions.h", directory: "/Stk/Ableton/img/Live/master")
!21 = !DISubprogram(name: "GetOption", linkageName: "_ZN13ADebugOptions9GetOptionEPKc", scope: !19, file: !20, line: 133, type: !15, isLocal: false, isDefinition: false, scopeLine: 133, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!22 = !DILocation(line: 139, column: 42, scope: !23, inlinedAt: !30)
!23 = distinct !DISubprogram(name: "address", linkageName: "_ZN5boost15optional_detail15aligned_storageI12TDebugOptionE7addressEv", scope: !25, file: !24, line: 139, type: !15, isLocal: false, isDefinition: true, scopeLine: 139, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !29, variables: !2)
!24 = !DIFile(filename: "modules/boost/include/boost/optional/optional.hpp", directory: "/Stk/Ableton/img/Live/master")
!25 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "aligned_storage<TDebugOption>", scope: !26, file: !24, line: 120, size: 128, align: 64, elements: !2, templateParams: !2, identifier: "_ZTSN5boost15optional_detail15aligned_storageI12TDebugOptionEE")
!26 = !DINamespace(name: "optional_detail", scope: !27, file: !24, line: 114)
!27 = !DINamespace(name: "boost", scope: null, file: !28, line: 482)
!28 = !DIFile(filename: "modules/boost/include/boost/config/suffix.hpp", directory: "/Stk/Ableton/img/Live/master")
!29 = !DISubprogram(name: "address", linkageName: "_ZN5boost15optional_detail15aligned_storageI12TDebugOptionE7addressEv", scope: !25, file: !24, line: 139, type: !15, isLocal: false, isDefinition: false, scopeLine: 139, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!30 = distinct !DILocation(line: 728, column: 71, scope: !31, inlinedAt: !37)
!31 = distinct !DISubprogram(name: "get_object", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE10get_objectEv", scope: !32, file: !24, line: 726, type: !15, isLocal: false, isDefinition: true, scopeLine: 727, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !33, variables: !34)
!32 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "optional_base<TDebugOption>", scope: !26, file: !24, line: 197, size: 192, align: 64, elements: !2, templateParams: !2, identifier: "_ZTSN5boost15optional_detail13optional_baseI12TDebugOptionEE")
!33 = !DISubprogram(name: "get_object", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE10get_objectEv", scope: !32, file: !24, line: 726, type: !15, isLocal: false, isDefinition: false, scopeLine: 726, flags: DIFlagPrototyped, isOptimized: true)
!34 = !{!35}
!35 = !DILocalVariable(name: "caster", scope: !31, file: !24, line: 728, type: !36)
!36 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !31, file: !24, line: 728, size: 64, align: 64, elements: !2, identifier: "_ZTSZN5boost15optional_detail13optional_baseI12TDebugOptionE10get_objectEvEUt_")
!37 = distinct !DILocation(line: 711, column: 64, scope: !38, inlinedAt: !40)
!38 = distinct !DISubprogram(name: "get_impl", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE8get_implEv", scope: !32, file: !24, line: 711, type: !15, isLocal: false, isDefinition: true, scopeLine: 711, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !39, variables: !2)
!39 = !DISubprogram(name: "get_impl", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE8get_implEv", scope: !32, file: !24, line: 711, type: !15, isLocal: false, isDefinition: false, scopeLine: 711, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: true)
!40 = distinct !DILocation(line: 700, column: 75, scope: !41, inlinedAt: !43)
!41 = distinct !DISubprogram(name: "assign_value", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE12assign_valueEOS2_N4mpl_5bool_ILb0EEE", scope: !32, file: !24, line: 700, type: !15, isLocal: false, isDefinition: true, scopeLine: 700, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !42, variables: !2)
!42 = !DISubprogram(name: "assign_value", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE12assign_valueEOS2_N4mpl_5bool_ILb0EEE", scope: !32, file: !24, line: 700, type: !15, isLocal: false, isDefinition: false, scopeLine: 700, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: true)
!43 = distinct !DILocation(line: 422, column: 12, scope: !44, inlinedAt: !47)
!44 = distinct !DILexicalBlock(scope: !45, file: !24, line: 421, column: 11)
!45 = distinct !DISubprogram(name: "assign", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE6assignEOS2_", scope: !32, file: !24, line: 419, type: !15, isLocal: false, isDefinition: true, scopeLine: 420, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !46, variables: !2)
!46 = !DISubprogram(name: "assign", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE6assignEOS2_", scope: !32, file: !24, line: 419, type: !15, isLocal: false, isDefinition: false, scopeLine: 419, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: true)
!47 = distinct !DILocation(line: 961, column: 9, scope: !48, inlinedAt: !51)
!48 = distinct !DISubprogram(name: "operator=", linkageName: "_ZN5boost8optionalI12TDebugOptionEaSEOS1_", scope: !49, file: !24, line: 958, type: !15, isLocal: false, isDefinition: true, scopeLine: 959, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !50, variables: !2)
!49 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "optional<TDebugOption>", scope: !27, file: !24, line: 765, size: 192, align: 64, elements: !2, templateParams: !2, identifier: "_ZTSN5boost8optionalI12TDebugOptionEE")
!50 = !DISubprogram(name: "operator=", linkageName: "_ZN5boost8optionalI12TDebugOptionEaSEOS1_", scope: !49, file: !24, line: 958, type: !15, isLocal: false, isDefinition: false, scopeLine: 958, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!51 = distinct !DILocation(line: 795, column: 16, scope: !52)
!52 = distinct !DILexicalBlock(scope: !53, file: !13, line: 794, column: 7)
!53 = distinct !DILexicalBlock(scope: !54, file: !13, line: 793, column: 11)
!54 = distinct !DISubprogram(name: "operator()", linkageName: "_ZN12_GLOBAL__N_113TOptionGetterclERKNS_13TDebugOptionsE", scope: !55, file: !13, line: 783, type: !15, isLocal: true, isDefinition: true, scopeLine: 784, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !56, variables: !2)
!55 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TOptionGetter", scope: !5, file: !13, line: 776, size: 64, align: 64, elements: !2)
!56 = !DISubprogram(name: "operator()", linkageName: "_ZN12_GLOBAL__N_113TOptionGetterclERKNS_13TDebugOptionsE", scope: !55, file: !13, line: 783, type: !15, isLocal: false, isDefinition: false, scopeLine: 783, flags: DIFlagPrototyped, isOptimized: true)
!57 = !{i32 2, !"Debug Info Version", i32 700000003}
