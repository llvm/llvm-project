; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Generated from:
;
; struct A {
;   static int fully_specified;
;   static int smem[];
; };
;  
; int A::fully_specified;
; int A::smem[] = { 0, 1, 2, 3 };
;
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification{{.*}}"fully_specified"
; CHECK-NOT:  DW_AT_type
; CHECK: DW_TAG_structure_type
; CHECK: DW_TAG_member
; CHECK: DW_TAG_member
; CHECK-NEXT:   DW_AT_name {{.*}} "smem"
; CHECK-NEXT:   DW_AT_type {{.*}} {0x[[GENERIC:[0-9]+]]}
;
; CHECK: 0x[[GENERIC]]: DW_TAG_array_type
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  NULL
; CHECK: DW_TAG_subrange_type
; CHECK-NOT:  DW_AT_count
; CHECK:  NULL
;
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}}"smem"
; CHECK-NEXT: DW_AT_type {{.*}} {0x[[SPECIFIC:[0-9]+]]}
;
; CHECK: 0x[[SPECIFIC]]: DW_TAG_array_type
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  NULL
; CHECK: DW_TAG_subrange_type
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  NULL
; CHECK:  DW_AT_count {{.*}} (0x04)

source_filename = "static_member_array.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@_ZN1A15fully_specifiedE = global i32 0, align 4, !dbg !0
@_ZN1A4smemE = global [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16, !dbg !5

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = distinct !DIGlobalVariable(name: "fully_specified", linkageName: "_ZN1A15fully_specifiedE", scope: !1, file: !2, line: 7, type: !7, isLocal: false, isDefinition: true, declaration: !13)
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 4.0.0 (trunk 286129) (llvm/trunk 286128)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "static_member_array.cpp", directory: "/Volumes/Data/radar/28706946")
!3 = !{}
!4 = !{!0, !5}
!5 = distinct !DIGlobalVariable(name: "smem", linkageName: "_ZN1A4smemE", scope: !1, file: !2, line: 8, type: !6, isLocal: false, isDefinition: true, declaration: !10)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !8)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 4)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "smem", scope: !11, file: !2, line: 4, baseType: !14, flags: DIFlagStaticMember)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !2, line: 1, size: 8, elements: !12, identifier: "_ZTS1A")
!12 = !{!13, !10}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "fully_specified", scope: !11, file: !2, line: 3, baseType: !7, flags: DIFlagStaticMember)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: -1)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"PIC Level", i32 2}
!20 = !{!"clang version 4.0.0 (trunk 286129) (llvm/trunk 286128)"}
