
; RUN: llc -split-dwarf-file=foo.dwo -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s

; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck --check-prefix=SINGLE %s

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

source_filename = "test/DebugInfo/X86/type_units_with_addresses.ll"

%struct.S1 = type { i8 }
%struct.S2 = type { %struct.S2_1 }
%struct.S2_1 = type { i8 }
%struct.S3 = type { %struct.S3_1, %struct.S3_2 }
%struct.S3_1 = type { i8 }
%struct.S3_2 = type { i8 }
%struct.S4 = type { %struct.S4_1, %struct.S4_2 }
%struct.S4_1 = type { i8 }
%struct.S4_2 = type { i8 }

@i = global i32 0, align 4, !dbg !0
@a = global %struct.S1 zeroinitializer, align 1, !dbg !4
@s2 = global %struct.S2 zeroinitializer, align 1, !dbg !11
@s3 = global %struct.S3 zeroinitializer, align 1, !dbg !17
@s4 = global %struct.S4 zeroinitializer, align 1, !dbg !25

!llvm.dbg.cu = !{!35}
!llvm.module.flags = !{!38, !39}
!llvm.ident = !{!40}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "i", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "tu.cpp", directory: "/tmp/dbginfo")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 6, type: !6, isLocal: false, isDefinition: true)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "S1<&i>", file: !2, line: 4, size: 8, align: 8, elements: !7, templateParams: !8, identifier: "_ZTS2S1IXadL_Z1iEEE")
!7 = !{}
!8 = !{!9}
!9 = !DITemplateValueParameter(name: "I", type: !10, value: ptr @i)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = !DIGlobalVariable(name: "s2", scope: null, file: !2, line: 15, type: !13, isLocal: false, isDefinition: true)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "S2", file: !2, line: 11, size: 8, align: 8, elements: !14, identifier: "_ZTS2S2")
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "s2_1", scope: !13, file: !2, line: 12, baseType: !16, size: 8, align: 8)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "S2_1<&i>", file: !2, line: 9, size: 8, align: 8, elements: !7, templateParams: !8, identifier: "_ZTS4S2_1IXadL_Z1iEEE")
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = !DIGlobalVariable(name: "s3", scope: null, file: !2, line: 27, type: !19, isLocal: false, isDefinition: true)
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "S3", file: !2, line: 22, size: 16, align: 8, elements: !20, identifier: "_ZTS2S3")
!20 = !{!21, !23}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "s3_1", scope: !19, file: !2, line: 23, baseType: !22, size: 8, align: 8)
!22 = !DICompositeType(tag: DW_TAG_structure_type, name: "S3_1<&i>", file: !2, line: 18, size: 8, align: 8, elements: !7, templateParams: !8, identifier: "_ZTS4S3_1IXadL_Z1iEEE")
!23 = !DIDerivedType(tag: DW_TAG_member, name: "s3_2", scope: !19, file: !2, line: 24, baseType: !24, size: 8, align: 8, offset: 8)
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "S3_2", file: !2, line: 20, size: 8, align: 8, elements: !7, identifier: "_ZTS4S3_2")
!25 = !DIGlobalVariableExpression(var: !26, expr: !DIExpression())
!26 = !DIGlobalVariable(name: "s4", scope: null, file: !2, line: 39, type: !27, isLocal: false, isDefinition: true)
!27 = !DICompositeType(tag: DW_TAG_structure_type, name: "S4", file: !2, line: 34, size: 16, align: 8, elements: !28, identifier: "_ZTS2S4")
!28 = !{!29, !31}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "s4_1", scope: !27, file: !2, line: 35, baseType: !30, size: 8, align: 8)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "S4_1", file: !2, line: 29, size: 8, align: 8, elements: !7, identifier: "_ZTS4S4_1")
!31 = !DIDerivedType(tag: DW_TAG_member, name: "s4_2", scope: !27, file: !2, line: 36, baseType: !32, size: 8, align: 8, offset: 8)
!32 = !DICompositeType(tag: DW_TAG_structure_type, name: "S4_2<&i>", file: !2, line: 32, size: 8, align: 8, elements: !7, templateParams: !33, identifier: "_ZTS4S4_2IXadL_Z1iEEE")
!33 = !{!34}
!34 = !DITemplateValueParameter(name: "T", type: !10, value: ptr @i)
!35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "tu.dwo", emissionKind: FullDebug, enums: !7, retainedTypes: !36, globals: !37, imports: !7)
!36 = !{!6, !13, !16, !19, !22, !24, !27, !30, !32}
!37 = !{!0, !4, !11, !17, !25}
!38 = !{i32 2, !"Dwarf Version", i32 4}
!39 = !{i32 1, !"Debug Info Version", i32 3}
!40 = !{!"clang version 3.5.0 "}

