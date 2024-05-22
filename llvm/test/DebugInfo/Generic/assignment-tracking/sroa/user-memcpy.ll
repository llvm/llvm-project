; RUN: opt -passes=sroa -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators -passes=sroa -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Check that the fragments generated in SROA for a split alloca that has a
;; dbg.assign with non-zero-offset fragment are correct.

;; $ cat test.cpp
;; #include <cstring>
;;
;; struct V3i { long x, y, z; };
;; void fun() {
;;   V3i point = {0, 0, 0};
;;   point.z = 5000;
;;   V3i other = {10, 9, 8};
;;   std::memcpy(&point.y, &other.x, sizeof(long) * 2);
;; }
;; $ clang++ -c -O2 -g test.cpp -o - -Xclang -disable-llvm-passes -S -emit-llvm \
;;   | opt -passes=declare-to-assign -S -o -

; CHECK: entry:
;; Allocas have been promoted - the linked dbg.assigns have been removed.

;; | V3i point = {0, 0, 0};
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 0, metadata ![[point:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64))
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 0, metadata ![[point]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64))

;; point.z = 5000;
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 5000, metadata ![[point]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64))

;; | V3i other = {10, 9, 8};
;;   other is global const:
;;     local.other.x = global.other.x
;;     local.other.y = global.other.y
;;     local.other.z = global.other.z
; CHECK-NEXT: %other.sroa.0.0.copyload = load i64, ptr @__const._Z3funv.other
; CHECK-NEXT: %other.sroa.2.0.copyload = load i64, ptr getelementptr inbounds (i8, ptr @__const._Z3funv.other, i64 8)
; CHECK-NEXT: %other.sroa.3.0.copyload = load i64, ptr getelementptr inbounds (i8, ptr @__const._Z3funv.other, i64 16)
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 %other.sroa.0.0.copyload, metadata ![[other:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64))
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 %other.sroa.2.0.copyload, metadata ![[other]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64))
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 %other.sroa.3.0.copyload, metadata ![[other]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64))

;; | std::memcpy(&point.y, &other.x, sizeof(long) * 2);
;;   other is now 3 scalars:
;;     point.y = other.x
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 %other.sroa.0.0.copyload, metadata ![[point]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64))
;;
;;     point.z = other.y
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 %other.sroa.2.0.copyload, metadata ![[point]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64))

; CHECK: ![[point]] = !DILocalVariable(name: "point",
; CHECK: ![[other]] = !DILocalVariable(name: "other",

source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.V3i = type { i64, i64, i64 }

@__const._Z3funv.other = private unnamed_addr constant %struct.V3i { i64 10, i64 9, i64 8 }, align 8

; Function Attrs: nounwind uwtable mustprogress
define dso_local void @_Z3funv() !dbg !100 {
entry:
  %point = alloca %struct.V3i, align 8, !DIAssignID !112
  call void @llvm.dbg.assign(metadata i1 undef, metadata !104, metadata !DIExpression(), metadata !112, metadata ptr %point, metadata !DIExpression()), !dbg !113
  %other = alloca %struct.V3i, align 8, !DIAssignID !114
  call void @llvm.dbg.assign(metadata i1 undef, metadata !111, metadata !DIExpression(), metadata !114, metadata ptr %other, metadata !DIExpression()), !dbg !113
  %0 = bitcast ptr %point to ptr, !dbg !115
  %1 = bitcast ptr %point to ptr, !dbg !116
  call void @llvm.memset.p0.i64(ptr align 8 %1, i8 0, i64 24, i1 false), !dbg !116, !DIAssignID !117
  call void @llvm.dbg.assign(metadata i8 0, metadata !104, metadata !DIExpression(), metadata !117, metadata ptr %1, metadata !DIExpression()), !dbg !116
  %z = getelementptr inbounds %struct.V3i, ptr %point, i32 0, i32 2, !dbg !118
  store i64 5000, ptr %z, align 8, !dbg !119, !DIAssignID !125
  call void @llvm.dbg.assign(metadata i64 5000, metadata !104, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64), metadata !125, metadata ptr %z, metadata !DIExpression()), !dbg !119
  %2 = bitcast ptr %other to ptr, !dbg !126
  %3 = bitcast ptr %other to ptr, !dbg !127
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %3, ptr align 8 @__const._Z3funv.other, i64 24, i1 false), !dbg !127, !DIAssignID !128
  call void @llvm.dbg.assign(metadata i1 undef, metadata !111, metadata !DIExpression(), metadata !128, metadata ptr %3, metadata !DIExpression()), !dbg !127
  %y = getelementptr inbounds %struct.V3i, ptr %point, i32 0, i32 1, !dbg !129
  %4 = bitcast ptr %y to ptr, !dbg !130
  %x = getelementptr inbounds %struct.V3i, ptr %other, i32 0, i32 0, !dbg !131
  %5 = bitcast ptr %x to ptr, !dbg !130
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %4, ptr align 8 %5, i64 16, i1 false), !dbg !130, !DIAssignID !132
  call void @llvm.dbg.assign(metadata i1 undef, metadata !104, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 128), metadata !132, metadata ptr %4, metadata !DIExpression()), !dbg !130
  %6 = bitcast ptr %other to ptr, !dbg !133
  %7 = bitcast ptr %point to ptr, !dbg !133
  ret void, !dbg !133
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!96, !97, !98, !1000}
!llvm.ident = !{!99}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{!4, !18, !22, !28, !32, !36, !46, !50, !52, !54, !58, !62, !66, !70, !74, !76, !78, !80, !84, !88, !92, !94}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !6, file: !17, line: 75)
!5 = !DINamespace(name: "std", scope: null)
!6 = !DISubprogram(name: "memchr", scope: !7, file: !7, line: 90, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!7 = !DIFile(filename: "/usr/include/string.h", directory: "")
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !13, !14}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !15, line: 46, baseType: !16)
!15 = !DIFile(filename: "lib/clang/12.0.0/include/stddef.h", directory: "/")
!16 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!17 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/cstring", directory: "")
!18 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !19, file: !17, line: 76)
!19 = !DISubprogram(name: "memcmp", scope: !7, file: !7, line: 63, type: !20, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!20 = !DISubroutineType(types: !21)
!21 = !{!13, !11, !11, !14}
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !23, file: !17, line: 77)
!23 = !DISubprogram(name: "memcpy", scope: !7, file: !7, line: 42, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!24 = !DISubroutineType(types: !25)
!25 = !{!10, !26, !27, !14}
!26 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !10)
!27 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !11)
!28 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !29, file: !17, line: 78)
!29 = !DISubprogram(name: "memmove", scope: !7, file: !7, line: 46, type: !30, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!30 = !DISubroutineType(types: !31)
!31 = !{!10, !10, !11, !14}
!32 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !33, file: !17, line: 79)
!33 = !DISubprogram(name: "memset", scope: !7, file: !7, line: 60, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!34 = !DISubroutineType(types: !35)
!35 = !{!10, !10, !13, !14}
!36 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !37, file: !17, line: 80)
!37 = !DISubprogram(name: "strcat", scope: !7, file: !7, line: 129, type: !38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!38 = !DISubroutineType(types: !39)
!39 = !{!40, !42, !43}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !41, size: 64)
!41 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!42 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !40)
!43 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !44)
!44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !45, size: 64)
!45 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !41)
!46 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !47, file: !17, line: 81)
!47 = !DISubprogram(name: "strcmp", scope: !7, file: !7, line: 136, type: !48, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!48 = !DISubroutineType(types: !49)
!49 = !{!13, !44, !44}
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !51, file: !17, line: 82)
!51 = !DISubprogram(name: "strcoll", scope: !7, file: !7, line: 143, type: !48, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!52 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !53, file: !17, line: 83)
!53 = !DISubprogram(name: "strcpy", scope: !7, file: !7, line: 121, type: !38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !55, file: !17, line: 84)
!55 = !DISubprogram(name: "strcspn", scope: !7, file: !7, line: 272, type: !56, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!56 = !DISubroutineType(types: !57)
!57 = !{!14, !44, !44}
!58 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !59, file: !17, line: 85)
!59 = !DISubprogram(name: "strerror", scope: !7, file: !7, line: 396, type: !60, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!60 = !DISubroutineType(types: !61)
!61 = !{!40, !13}
!62 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !63, file: !17, line: 86)
!63 = !DISubprogram(name: "strlen", scope: !7, file: !7, line: 384, type: !64, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!64 = !DISubroutineType(types: !65)
!65 = !{!14, !44}
!66 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !67, file: !17, line: 87)
!67 = !DISubprogram(name: "strncat", scope: !7, file: !7, line: 132, type: !68, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!68 = !DISubroutineType(types: !69)
!69 = !{!40, !42, !43, !14}
!70 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !71, file: !17, line: 88)
!71 = !DISubprogram(name: "strncmp", scope: !7, file: !7, line: 139, type: !72, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!72 = !DISubroutineType(types: !73)
!73 = !{!13, !44, !44, !14}
!74 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !75, file: !17, line: 89)
!75 = !DISubprogram(name: "strncpy", scope: !7, file: !7, line: 124, type: !68, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!76 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !77, file: !17, line: 90)
!77 = !DISubprogram(name: "strspn", scope: !7, file: !7, line: 276, type: !56, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!78 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !79, file: !17, line: 91)
!79 = !DISubprogram(name: "strtok", scope: !7, file: !7, line: 335, type: !38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!80 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !81, file: !17, line: 92)
!81 = !DISubprogram(name: "strxfrm", scope: !7, file: !7, line: 146, type: !82, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!82 = !DISubroutineType(types: !83)
!83 = !{!14, !42, !43, !14}
!84 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !85, file: !17, line: 93)
!85 = !DISubprogram(name: "strchr", scope: !7, file: !7, line: 225, type: !86, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!86 = !DISubroutineType(types: !87)
!87 = !{!40, !44, !13}
!88 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !89, file: !17, line: 94)
!89 = !DISubprogram(name: "strpbrk", scope: !7, file: !7, line: 302, type: !90, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!90 = !DISubroutineType(types: !91)
!91 = !{!40, !44, !44}
!92 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !93, file: !17, line: 95)
!93 = !DISubprogram(name: "strrchr", scope: !7, file: !7, line: 252, type: !86, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!94 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !95, file: !17, line: 96)
!95 = !DISubprogram(name: "strstr", scope: !7, file: !7, line: 329, type: !90, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!96 = !{i32 7, !"Dwarf Version", i32 4}
!97 = !{i32 2, !"Debug Info Version", i32 3}
!98 = !{i32 1, !"wchar_size", i32 4}
!99 = !{!"clang version 12.0.0"}
!100 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 4, type: !101, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !103)
!101 = !DISubroutineType(types: !102)
!102 = !{null}
!103 = !{!104, !111}
!104 = !DILocalVariable(name: "point", scope: !100, file: !1, line: 5, type: !105)
!105 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "V3i", file: !1, line: 3, size: 192, flags: DIFlagTypePassByValue, elements: !106, identifier: "_ZTS3V3i")
!106 = !{!107, !109, !110}
!107 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !105, file: !1, line: 3, baseType: !108, size: 64)
!108 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!109 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !105, file: !1, line: 3, baseType: !108, size: 64, offset: 64)
!110 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !105, file: !1, line: 3, baseType: !108, size: 64, offset: 128)
!111 = !DILocalVariable(name: "other", scope: !100, file: !1, line: 7, type: !105)
!112 = distinct !DIAssignID()
!113 = !DILocation(line: 0, scope: !100)
!114 = distinct !DIAssignID()
!115 = !DILocation(line: 5, column: 3, scope: !100)
!116 = !DILocation(line: 5, column: 7, scope: !100)
!117 = distinct !DIAssignID()
!118 = !DILocation(line: 6, column: 9, scope: !100)
!119 = !DILocation(line: 6, column: 11, scope: !100)
!125 = distinct !DIAssignID()
!126 = !DILocation(line: 7, column: 3, scope: !100)
!127 = !DILocation(line: 7, column: 7, scope: !100)
!128 = distinct !DIAssignID()
!129 = !DILocation(line: 8, column: 22, scope: !100)
!130 = !DILocation(line: 8, column: 3, scope: !100)
!131 = !DILocation(line: 8, column: 32, scope: !100)
!132 = distinct !DIAssignID()
!133 = !DILocation(line: 9, column: 1, scope: !100)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
