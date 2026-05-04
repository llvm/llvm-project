; REQUIRES: target={{x86_64-.*-linux.*}}
; RUN: llc -O0 -filetype=obj < %s |   \
; RUN: llvm-dwarfdump --debug-info - | FileCheck %s

; LLVM IR generated from:

; struct S {
;   static int Member;   <-- Manually marked as artificial
; };
; int S::Member = 1;

source_filename = "artificial-static-member.cpp"
target triple = "x86_64-pc-linux-gnu"

@_ZN1S6MemberE = dso_local global i32 1, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Member", linkageName: "_ZN1S6MemberE", scope: !2, type: !5, isLocal: false, isDefinition: true, declaration: !6)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "artificial-static-member.cpp", directory: "")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_variable, name: "Member", scope: !7, baseType: !5, flags: DIFlagArtificial | DIFlagStaticMember)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", size: 8, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS1S")
!8 = !{!6}
!9 = !{i32 7, !"Dwarf Version", i32 5}
!10 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: {{.*}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_calling_convention
; CHECK-NEXT: DW_AT_name	("S")
; CHECK-NEXT: DW_AT_byte_size

; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name	("Member")
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_external	(true)
; CHECK-NEXT: DW_AT_declaration	(true)
; CHECK-NEXT: DW_AT_artificial	(true)
