; RUN: %llc_dwarf -filetype=obj %s -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; A hand-written testcase to check 64-bit constant handling in location lists.

; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location [DW_FORM_data4]	(
; CHECK-NEXT:   {{.*}}: DW_OP_constu 0x4000000000000000)
; CHECK-NEXT: DW_AT_name {{.*}}"d"
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location [DW_FORM_data4]	(
; CHECK-NEXT:   {{.*}}: DW_OP_consts +0
; CHECK-NEXT:   {{.*}}: DW_OP_consts +4611686018427387904)
; CHECK-NEXT: DW_AT_name {{.*}}"i"
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location [DW_FORM_data4]	(
; CHECK-NEXT:   {{.*}}: DW_OP_lit0
; CHECK-NEXT:   {{.*}}: DW_OP_constu 0x4000000000000000)
; CHECK-NEXT: DW_AT_name {{.*}}"u"

source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; Function Attrs: nounwind ssp uwtable
define void @main() #0 !dbg !7 {
  %1 = alloca double, align 8
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store double 2.000000e+00, double* %1, align 8, !dbg !21
  call void @llvm.dbg.value(metadata i64 0, metadata !22, metadata !15), !dbg !24
  call void @llvm.dbg.value(metadata i64 0, metadata !25, metadata !15), !dbg !27
  call void @llvm.dbg.value(metadata double 2.000000e+00, metadata !19, metadata !15), !dbg !21
  store i64 4611686018427387904, i64* %2, align 8, !dbg !24
  call void @llvm.dbg.value(metadata i64 4611686018427387904, metadata !22, metadata !15), !dbg !24
  call void @llvm.dbg.value(metadata i64 4611686018427387904, metadata !25, metadata !15), !dbg !27
  store i64 4611686018427387904, i64* %3, align 8, !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #1


attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 00000003}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !{})
!15 = !DIExpression()
!16 = !DILocation(line: 1, column: 14, scope: !7)
!18 = !DILocation(line: 1, column: 24, scope: !7)
!19 = !DILocalVariable(name: "d", scope: !7, file: !1, line: 2, type: !20)
!20 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!21 = !DILocation(line: 2, column: 10, scope: !7)
!22 = !DILocalVariable(name: "u", scope: !7, file: !1, line: 3, type: !23)
!23 = !DIBasicType(name: "long long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!24 = !DILocation(line: 3, column: 22, scope: !7)
!25 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 4, type: !26)
!26 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!27 = !DILocation(line: 4, column: 20, scope: !7)
!28 = !DILocation(line: 5, column: 3, scope: !7)
