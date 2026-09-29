; RUN: llc < %s -o - | FileCheck %s

; Check that no assert failure occurs when there's a funclet in a swifttailcc function.
; CHECK-LABEL: foo:

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.38.33134"

declare ptr @F2()

declare void @F3(ptr)

declare i32 @__CxxFrameHandler3(...)

define swifttailcc void @foo(ptr swiftasync %0, ptr %v3, ptr %v4) personality ptr @__CxxFrameHandler3 !dbg !88 {
v21:
  call void @F3(ptr %0)
  %v28 = invoke ptr @F2(ptr %v3, ptr %v4)
          to label %v29 unwind label %v39

v29:                                              ; preds = %v21
  ret void

v39:                                              ; preds = %v21
  %v40 = cleanuppad within none []
  %v43 = load i1, ptr null, align 1
  cleanupret from %v40 unwind to caller
}

; This test is sensitive to the debug info

!llvm.dbg.cu = !{!0, !11}
!llvm.module.flags = !{!87}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift version 5.11-dev", isOptimized: true, flags: "-private-discriminator _0FA2C3DB08FF346F3CA00B2EC660E1DF -enable-experimental-cxx-interop", runtimeVersion: 5, emissionKind: FullDebug, globals: !2, imports: !10, sysroot: "S:\\Windows.sdk", sdk: "Windows.sdk")
!1 = !DIFile(filename: "foo.swift", directory: "C:\\foo")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(name: "mutationQueue", linkageName: "$sSo8fire", scope: !5, file: !1, line: 18, type: !6, isLocal: false, isDefinition: true)
!5 = !DIModule(scope: null, name: "FF", includePath: "C:\\foo")
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "DispatchQueue", scope: !9, file: !8, size: 64, elements: !10, runtimeLang: DW_LANG_Swift, identifier: "$s8Dispatch0A5QueueCD")
!8 = !DIFile(filename: "S:\\aarch64-unknown-windows-msvc.swiftmodule", directory: "")
!9 = !DIModule(scope: null, name: "Dispatch", configMacros: "\22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "S:\\aarch64-unknown-windows-msvc.swiftmodule")
!10 = !{}
!11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !12, producer: "clang version 17.0.6", isOptimized: true, flags: "-private-discriminator _0FA2C3DB08FF346F3CA00B2EC660E1DF -enable-experimental-cxx-interop", runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !10, globals: !13, splitDebugInlining: false, nameTableKind: None)
!12 = !DIFile(filename: "<swift-imported-modules>", directory: "C:\\foo")
!13 = !{!14, !20, !29, !36, !40, !43, !53, !55, !59, !62, !77}
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = distinct !DIGlobalVariable(scope: null, file: !16, line: 2314, type: !17, isLocal: true, isDefinition: true)
!16 = !DIFile(filename: "C:\\xstring", directory: "")
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 128, elements: !10)
!18 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !19)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "_Fake_alloc", linkageName: "?_Fake_alloc@std@@3U_Fake_allocator@1@B", scope: !22, file: !26, line: 1491, type: !27, isLocal: true, isDefinition: true)
!22 = !DINamespace(name: "std", scope: !23)
!23 = !DIModule(scope: !24, name: "xmemory", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!24 = !DIModule(scope: !25, name: "_Private", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!25 = !DIModule(scope: null, name: "std", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!26 = !DIFile(filename: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\\xmemory", directory: "")
!27 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !28)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_Fake_allocator", scope: !22, file: !26, line: 1169, size: 8, flags: DIFlagFwdDecl, identifier: ".?AU_Fake_allocator@std@@")
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression(DW_OP_constu, 8, DW_OP_stack_value))
!30 = distinct !DIGlobalVariable(name: "_Small_object_num_ptrs", scope: !31, file: !33, line: 28, type: !34, isLocal: true, isDefinition: true)
!31 = !DINamespace(name: "std", scope: !32)
!32 = !DIModule(scope: !25, name: "typeinfo", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!33 = !DIFile(filename: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\\typeinfo", directory: "")
!34 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !35)
!35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!36 = !DIGlobalVariableExpression(var: !37, expr: !DIExpression())
!37 = distinct !DIGlobalVariable(scope: null, file: !38, line: 140, type: !39, isLocal: true, isDefinition: true)
!38 = !DIFile(filename: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\\vcruntime_exception.h", directory: "")
!39 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 168, elements: !10)
!40 = !DIGlobalVariableExpression(var: !41, expr: !DIExpression())
!41 = distinct !DIGlobalVariable(scope: null, file: !38, line: 95, type: !42, isLocal: true, isDefinition: true)
!42 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 144, elements: !10)
!43 = !DIGlobalVariableExpression(var: !44, expr: !DIExpression(DW_OP_constu, 14695981039346656037, DW_OP_stack_value))
!44 = distinct !DIGlobalVariable(name: "_FNV_offset_basis", scope: !45, file: !47, line: 2312, type: !48, isLocal: true, isDefinition: true)
!45 = !DINamespace(name: "std", scope: !46)
!46 = !DIModule(scope: !25, name: "type_traits", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!47 = !DIFile(filename: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\\type_traits", directory: "")
!48 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !49)
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", scope: !51, file: !50, line: 193, baseType: !52)
!50 = !DIFile(filename: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\\vcruntime.h", directory: "")
!51 = !DIModule(scope: null, name: "vcruntime", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!52 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!53 = !DIGlobalVariableExpression(var: !54, expr: !DIExpression(DW_OP_constu, 1099511628211, DW_OP_stack_value))
!54 = distinct !DIGlobalVariable(name: "_FNV_prime", scope: !45, file: !47, line: 2313, type: !48, isLocal: true, isDefinition: true)
!55 = !DIGlobalVariableExpression(var: !56, expr: !DIExpression())
!56 = distinct !DIGlobalVariable(scope: null, file: !57, line: 1645, type: !58, isLocal: true, isDefinition: true)
!57 = !DIFile(filename: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\\xhash", directory: "")
!58 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 216, elements: !10)
!59 = !DIGlobalVariableExpression(var: !60, expr: !DIExpression())
!60 = distinct !DIGlobalVariable(scope: null, file: !57, line: 1715, type: !61, isLocal: true, isDefinition: true)
!61 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 208, elements: !10)
!62 = !DIGlobalVariableExpression(var: !63, expr: !DIExpression())
!63 = distinct !DIGlobalVariable(name: "_Min_buckets", linkageName: "_Min_buckets", scope: !64, file: !57, line: 365, type: !65, isLocal: false, isDefinition: true, declaration: !76)
!64 = !DIModule(scope: !24, name: "xhash", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!65 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !66)
!66 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_type", scope: !67, file: !57, line: 347, baseType: !69, flags: DIFlagPublic)
!67 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "_Hash", scope: !68, file: !57, line: 330, size: 512, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !10, templateParams: !10, identifier: ".?AV?")
!68 = !DINamespace(name: "std", scope: !64)
!69 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_type", scope: !71, file: !70, line: 783, baseType: !74, flags: DIFlagPublic)
!70 = !DIFile(filename: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\\list", directory: "")
!71 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "list", scope: !72, file: !70, line: 755, size: 128, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !10, templateParams: !10, identifier: ".?AV?")
!72 = !DINamespace(name: "std", scope: !73)
!73 = !DIModule(scope: !25, name: "list", configMacros: "\22-D__swift__=51100\22 \22-D_ARM64_\22 \22-DINTERNAL_EXPERIMENTAL\22 \22-DSR69711\22", includePath: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include")
!74 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_type", scope: !75, file: !26, line: 656, baseType: !49)
!75 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_Default", scope: !22, file: !26, line: 647, size: 8, flags: DIFlagTypePassByValue, elements: !10, templateParams: !10, identifier: ".?AU?")
!76 = !DIDerivedType(tag: DW_TAG_member, name: "_Min_buckets", scope: !67, file: !57, line: 365, baseType: !65, flags: DIFlagPublic | DIFlagStaticMember, extraData: i64 8)
!77 = !DIGlobalVariableExpression(var: !78, expr: !DIExpression())
!78 = distinct !DIGlobalVariable(name: "_Min_buckets", linkageName: "_Min_buckets", scope: !64, file: !57, line: 365, type: !79, isLocal: false, isDefinition: true, declaration: !86)
!79 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !80)
!80 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_type", scope: !81, file: !57, line: 347, baseType: !82, flags: DIFlagPublic)
!81 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "_Hash", scope: !68, file: !57, line: 330, size: 512, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !10, templateParams: !10, identifier: ".?AV?")
!82 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_type", scope: !83, file: !70, line: 783, baseType: !84, flags: DIFlagPublic)
!83 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "list", scope: !72, file: !70, line: 755, size: 128, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !10, templateParams: !10, identifier: ".?AV?")
!84 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_type", scope: !85, file: !26, line: 656, baseType: !49)
!85 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_Default_allocator_traits", scope: !22, file: !26, line: 647, size: 8, flags: DIFlagTypePassByValue, elements: !10, templateParams: !10, identifier: ".?AU?")
!86 = !DIDerivedType(tag: DW_TAG_member, name: "_Min_buckets", scope: !81, file: !57, line: 365, baseType: !79, flags: DIFlagPublic | DIFlagStaticMember, extraData: i64 8)
!87 = !{i32 2, !"Debug Info Version", i32 3}
!88 = distinct !DISubprogram(name: "get", linkageName: "foo", scope: !5, file: !1, line: 32, type: !89, scopeLine: 32, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !90, retainedNodes: !10, thrownTypes: !10)
!89 = !DISubroutineType(types: !10)
!90 = !DISubprogram(name: "get", linkageName: "foo", scope: !5, file: !1, line: 32, type: !89, scopeLine: 32, spFlags: DISPFlagOptimized, thrownTypes: !10)
