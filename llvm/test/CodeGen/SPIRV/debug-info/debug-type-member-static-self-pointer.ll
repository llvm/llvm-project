; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=DebugTypeMember --implicit-check-not=DebugGlobalVariable

; struct S { static S *inst; }; S *S::inst;
; inst points back at S, so S is emitted without inst. The member must not be
; created for the global either.

; CHECK: DebugTypeComposite
; CHECK: DebugTypePointer

target triple = "spirv64-unknown-unknown"

@inst = addrspace(1) global ptr addrspace(1) null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "inst", scope: !2, file: !4, type: !7, isDefinition: true, declaration: !6)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, emissionKind: FullDebug, globals: !5)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIFile(filename: "t.cpp", directory: "/")
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "inst", scope: !8, file: !4, baseType: !7, flags: DIFlagStaticMember)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, dwarfAddressSpace: 1)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !4, size: 8, elements: !9)
!9 = !{!6}
