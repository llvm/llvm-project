; REQUIRES: object-emission

; RUN: llc -split-dwarf=Enable -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump - | FileCheck %s

; RUN: llc -split-dwarf=Disable -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump - | FileCheck --check-prefix=SINGLE %s

; Test case built from:
;int i;
;
;template <int *I>
;struct S1 {};
;
;S1<&i> s1;
;
;template <int *I>
;struct S2_1 {};
;
;struct S2 {
;  S2_1<&i> s2_1;
;};
;
;S2 s2;
;
;template <int *I>
;struct S3_1 {};
;
;struct S3_2 {};
;
;struct S3 {
;  S3_1<&i> s3_1;
;  S3_2 s3_2;
;};
;
;S3 s3;
;
;struct S4_1 {};
;
;template <int *T>
;struct S4_2 {};
;
;struct S4 {
;  S4_1 s4_1;
;  S4_2<&::i> s4_2;
;};
;
;S4 s4;


; CHECK: .debug_info.dwo contents:

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S1<&i>"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S2"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S2_1<&i>"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S3"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S3_1<&i>"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S4"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S4_2<&i>"

; SINGLE: .debug_info contents:

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

%struct.S1 = type { i8 }
%struct.S2 = type { %struct.S2_1 }
%struct.S2_1 = type { i8 }
%struct.S3 = type { %struct.S3_1, %struct.S3_2 }
%struct.S3_1 = type { i8 }
%struct.S3_2 = type { i8 }
%struct.S4 = type { %struct.S4_1, %struct.S4_2 }
%struct.S4_1 = type { i8 }
%struct.S4_2 = type { i8 }

@i = global i32 0, align 4
@a = global %struct.S1 zeroinitializer, align 1
@s2 = global %struct.S2 zeroinitializer, align 1
@s3 = global %struct.S3 zeroinitializer, align 1
@s4 = global %struct.S4 zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34, !35}
!llvm.ident = !{!36}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, splitDebugFilename: "tu.dwo", emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !27, imports: !2)
!1 = !DIFile(filename: "tu.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4, !9, !12, !13, !17, !18, !19, !23, !24}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "S1<&i>", line: 4, size: 8, align: 8, file: !1, elements: !2, templateParams: !5, identifier: "_ZTS2S1IXadL_Z1iEEE")
!5 = !{!6}
!6 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, name: "I", type: !7, value: i32* @i)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "S2", line: 11, size: 8, align: 8, file: !1, elements: !10, identifier: "_ZTS2S2")
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "s2_1", line: 12, size: 8, align: 8, file: !1, scope: !9, baseType: !12)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "S2_1<&i>", line: 9, size: 8, align: 8, file: !1, elements: !2, templateParams: !5, identifier: "_ZTS4S2_1IXadL_Z1iEEE")
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "S3", line: 22, size: 16, align: 8, file: !1, elements: !14, identifier: "_ZTS2S3")
!14 = !{!15, !16}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "s3_1", line: 23, size: 8, align: 8, file: !1, scope: !13, baseType: !17)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "s3_2", line: 24, size: 8, align: 8, offset: 8, file: !1, scope: !13, baseType: !18)
!17 = !DICompositeType(tag: DW_TAG_structure_type, name: "S3_1<&i>", line: 18, size: 8, align: 8, file: !1, elements: !2, templateParams: !5, identifier: "_ZTS4S3_1IXadL_Z1iEEE")
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "S3_2", line: 20, size: 8, align: 8, file: !1, elements: !2, identifier: "_ZTS4S3_2")
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "S4", line: 34, size: 16, align: 8, file: !1, elements: !20, identifier: "_ZTS2S4")
!20 = !{!21, !22}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "s4_1", line: 35, size: 8, align: 8, file: !1, scope: !19, baseType: !23)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "s4_2", line: 36, size: 8, align: 8, offset: 8, file: !1, scope: !19, baseType: !24)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "S4_1", line: 29, size: 8, align: 8, file: !1, elements: !2, identifier: "_ZTS4S4_1")
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "S4_2<&i>", line: 32, size: 8, align: 8, file: !1, elements: !2, templateParams: !25, identifier: "_ZTS4S4_2IXadL_Z1iEEE")
!25 = !{!26}
!26 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, name: "T", type: !7, value: i32* @i)
!27 = !{!28, !30, !31, !32, !33}
!28 = !DIGlobalVariable(name: "i", line: 1, isLocal: false, isDefinition: true, scope: null, file: !29, type: !8, variable: i32* @i)
!29 = !DIFile(filename: "tu.cpp", directory: "/tmp/dbginfo")
!30 = !DIGlobalVariable(name: "a", line: 6, isLocal: false, isDefinition: true, scope: null, file: !29, type: !4, variable: %struct.S1* @a)
!31 = !DIGlobalVariable(name: "s2", line: 15, isLocal: false, isDefinition: true, scope: null, file: !29, type: !9, variable: %struct.S2* @s2)
!32 = !DIGlobalVariable(name: "s3", line: 27, isLocal: false, isDefinition: true, scope: null, file: !29, type: !13, variable: %struct.S3* @s3)
!33 = !DIGlobalVariable(name: "s4", line: 39, isLocal: false, isDefinition: true, scope: null, file: !29, type: !19, variable: %struct.S4* @s4)
!34 = !{i32 2, !"Dwarf Version", i32 4}
!35 = !{i32 1, !"Debug Info Version", i32 3}
!36 = !{!"clang version 3.5.0 "}
