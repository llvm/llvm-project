; This test was obtained from swift source code and then automatically reducing it via Delta.
; The swift source code was from the test test/DebugInfo/debug_scope_distinct.swift.

; RUN: opt %s -S -p=sroa -o - | FileCheck %s

; CHECK: [[SROA_5_SROA_21:%.*]] = alloca [7 x i8], align 8
; CHECK-NEXT: #dbg_value(ptr [[SROA_5_SROA_21]], !59, !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 72, 56), [[DBG72:![0-9]+]])

; CHECK: #dbg_value(ptr [[REG1:%[0-9]+]], [[META54:![0-9]+]], !DIExpression(DW_OP_deref), [[DBG78:![0-9]+]])
; CHECK-NEXT: #dbg_value(ptr [[REG2:%[0-9]+]], [[META56:![0-9]+]], !DIExpression(DW_OP_deref), [[DBG78]])
; CHECK-NEXT: #dbg_value(i64 0, [[META57:![0-9]+]], !DIExpression(), [[DBG78]])

; CHECK: [[SROA_418_SROA_COPYLOAD:%.*]] = load i8, ptr [[SROA_418_0_U1_IDX:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT #dbg_value(i8 [[SROA_418_SROA_COPYLOAD]], [[META59]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 64, 8), [[DBG72]])

%T4main1TV13TangentVectorV = type <{ %T4main1UV13TangentVectorV, [7 x i8], %T4main1UV13TangentVectorV }>
%T4main1UV13TangentVectorV = type <{ %T1M1SVySfG, [7 x i8], %T4main1VV13TangentVectorV }>
%T1M1SVySfG = type <{ ptr, %Ts4Int8V }>
%Ts4Int8V = type <{ i8 }>
%T4main1VV13TangentVectorV = type <{ %T1M1SVySfG }>
define hidden swiftcc void @"$s4main1TV13TangentVectorV1poiyA2E_AEtFZ"(ptr noalias nocapture sret(%T4main1TV13TangentVectorV) %0, ptr noalias nocapture dereferenceable(57) %1, ptr noalias nocapture dereferenceable(57) %2) #0 !dbg !44 {
entry:
  %3 = alloca %T4main1VV13TangentVectorV
  %4 = alloca %T4main1UV13TangentVectorV
  call void @llvm.dbg.value(metadata ptr %1, metadata !54, metadata !DIExpression(DW_OP_deref)), !dbg !61
  call void @llvm.dbg.value(metadata ptr %2, metadata !56, metadata !DIExpression(DW_OP_deref)), !dbg !61
  call void @llvm.dbg.value(metadata i64 0, metadata !57, metadata !DIExpression()), !dbg !61
  %.u1 = getelementptr inbounds %T4main1TV13TangentVectorV, ptr %1, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %4, ptr align 8 %.u1, i64 25, i1 false), !dbg !61
  call void @llvm.dbg.value(metadata ptr %4, metadata !62, metadata !DIExpression(DW_OP_deref)), !dbg !75
  %.s = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %4, i32 0, i32 0
  %.s.b = getelementptr inbounds %T1M1SVySfG, ptr %.s, i32 0, i32 1
  %.s.b._value = getelementptr inbounds %Ts4Int8V, ptr %.s.b, i32 0, i32 0
  %12 = load i8, ptr %.s.b._value
  %.v = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %4, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %3, ptr align 8 %.v, i64 9, i1 false)
  %.s4 = getelementptr inbounds %T4main1VV13TangentVectorV, ptr %3, i32 0, i32 0
  %.s4.c = getelementptr inbounds %T1M1SVySfG, ptr %.s4, i32 0, i32 0
  %18 = load ptr, ptr %.s4.c
  ret void
}
!llvm.module.flags = !{!0, !1, !2, !3, !4, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!swift.module.flags = !{!33}
!llvm.linker.options = !{!34, !35, !36, !37, !38, !39, !40, !41, !42, !43}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 4]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,no_dead_strip"}
!4 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 1}
!11 = !{i32 7, !"frame-pointer", i32 1}
!12 = !{i32 1, !"Swift Version", i32 7}
!13 = !{i32 1, !"Swift ABI Version", i32 7}
!14 = !{i32 1, !"Swift Major Version", i8 6}
!15 = !{i32 1, !"Swift Minor Version", i8 0}
!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, imports: !18, sdk: "MacOSX14.4.sdk")
!17 = !DIFile(filename: "swift/swift/test/IRGen/debug_scope_distinct.swift", directory: "swift")
!18 = !{!19, !21, !23, !25, !27, !29, !31}
!19 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !20, file: !17)
!20 = !DIModule(scope: null, name: "main", includePath: "swift/swift/test/IRGen")
!21 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !22, file: !17)
!22 = !DIModule(scope: null, name: "Swift", includePath: "swift/_build/Ninja-RelWithDebInfoAssert+stdlib-RelWithDebInfo/swift-macosx-arm64/lib/swift/macosx/Swift.swiftmodule/arm64-apple-macos.swiftmodule")
!23 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !24, line: 60)
!24 = !DIModule(scope: null, name: "_Differentiation", includePath: "swift/_build/Ninja-RelWithDebInfoAssert+stdlib-RelWithDebInfo/swift-macosx-arm64/lib/swift/macosx/_Differentiation.swiftmodule/arm64-apple-macos.swiftmodule")
!25 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !26, line: 61)
!26 = !DIModule(scope: null, name: "M", includePath: "swift/_build/Ninja-RelWithDebInfoAssert+stdlib-RelWithDebInfo/swift-macosx-arm64/test-macosx-arm64/IRGen/Output/debug_scope_distinct.swift.tmp/M.swiftmodule")
!27 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !28, file: !17)
!28 = !DIModule(scope: null, name: "_StringProcessing", includePath: "swift/_build/Ninja-RelWithDebInfoAssert+stdlib-RelWithDebInfo/swift-macosx-arm64/lib/swift/macosx/_StringProcessing.swiftmodule/arm64-apple-macos.swiftmodule")
!29 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !30, file: !17)
!30 = !DIModule(scope: null, name: "_SwiftConcurrencyShims", includePath: "swift/_build/Ninja-RelWithDebInfoAssert+stdlib-RelWithDebInfo/swift-macosx-arm64/lib/swift/shims")
!31 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !32, file: !17)
!32 = !DIModule(scope: null, name: "_Concurrency", includePath: "swift/_build/Ninja-RelWithDebInfoAssert+stdlib-RelWithDebInfo/swift-macosx-arm64/lib/swift/macosx/_Concurrency.swiftmodule/arm64-apple-macos.swiftmodule")
!33 = !{ i1 false}
!34 = !{!"-lswiftCore"}
!35 = !{!"-lswift_StringProcessing"}
!36 = !{!"-lswift_Differentiation"}
!37 = !{!"-lswiftDarwin"}
!38 = !{!"-lswift_Concurrency"}
!39 = !{!"-lswiftSwiftOnoneSupport"}
!40 = !{!"-lobjc"}
!41 = !{!"-lswiftCompatibilityConcurrency"}
!42 = !{!"-lswiftCompatibility56"}
!43 = !{!"-lswiftCompatibilityPacks"}
!44 = distinct !DISubprogram(file: !45, type: !49, unit: !16, declaration: !52, retainedNodes: !53)
!45 = !DIFile(filename: "<compiler-generated>", directory: "/")
!46 = !DICompositeType(tag: DW_TAG_structure_type, scope: !47, elements: !48, identifier: "$s4main1TV13TangentVectorVD")
!47 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1TVD")
!48 = !{}
!49 = !DISubroutineType(types: !50)
!50 = !{ !51}
!51 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1TV13TangentVectorVXMtD")
!52 = !DISubprogram(spFlags: DISPFlagOptimized)
!53 = !{!54, !56, !57}
!54 = !DILocalVariable(name: "a", scope: !44, flags: DIFlagArtificial)
!55 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !46)
!56 = !DILocalVariable(name: "b", scope: !44, type: !55, flags: DIFlagArtificial)
!57 = !DILocalVariable(name: "c", scope: !44, type: !58, flags: DIFlagArtificial)
!58 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !51)
!61 = !DILocation(scope: !44)
!62 = !DILocalVariable(name: "d", scope: !63, type: !72, flags: DIFlagArtificial)
!63 = distinct !DISubprogram(unit: !16, retainedNodes: !70)
!64 = !DICompositeType(tag: DW_TAG_structure_type, size: 200, identifier: "$s4main1UV13TangentVectorVD")
!70 = !{}
!72 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !64)
!75 = !DILocation(scope: !63, inlinedAt: !76)
!76 = distinct !DILocation(scope: !44)
