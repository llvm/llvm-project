; RUN: llc -O0 -filetype=obj -o - %s | llvm-dwarfdump -v - | FileCheck %s
;
; CHECK: .debug_info contents:
; CHECK: [[LABEL_ORIGIN:0x[0-9a-zA-Z]+]]:{{ *}}DW_TAG_label
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] {{.*}}"top"
; CHECK-NEXT: DW_AT_decl_file [DW_FORM_data1] {{.*}}debug-label-inline.c
; CHECK-NEXT: DW_AT_decl_line [DW_FORM_data1] {{.*}}8
; CHECK: DW_TAG_label
; CHECK-NEXT: DW_AT_abstract_origin [DW_FORM_ref4] {{.*}}{[[LABEL_ORIGIN]]} "top"
; CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr] {{.*}}{{0x[0-9a-f]+}}

source_filename = "debug-label-inline.c"

@ga = external local_unnamed_addr global i32, align 4
@gb = external local_unnamed_addr global i32, align 4

define i32 @f2() local_unnamed_addr #0 !dbg !4 {
entry:
  %0 = load i32, i32* @ga, align 4, !dbg !1
  %1 = load i32, i32* @gb, align 4, !dbg !1
  call void @llvm.dbg.label(metadata !15), !dbg !17
  %add.i = add nsw i32 %1, %0, !dbg !18
  ret i32 %add.i, !dbg !1
}

declare void @llvm.dbg.label(metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { nounwind readonly }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, isOptimized: true, emissionKind: FullDebug, enums: !2)
!1 = !DILocation(line: 18, scope: !4)
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f2", scope: !6, file: !6, line: 15, type: !7, isLocal: false, isDefinition: true, scopeLine: 15, isOptimized: true, unit: !0, retainedNodes: !2)
!6 = !DIFile(filename: "debug-label-inline.c", directory: "./")
!7 = !DISubroutineType(types: !8)
!8 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DISubprogram(name: "f1", scope: !6, file: !6, line: 5, type: !12, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: true, unit: !0, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{!10, !10, !10}
!14 = !{!15}
!15 = !DILabel(scope: !11, name: "top", file: !6, line: 8)
!16 = distinct !DILocation(line: 18, scope: !4)
!17 = !DILocation(line: 8, scope: !11, inlinedAt: !16)
!18 = !DILocation(line: 9, scope: !11, inlinedAt: !16)
