; Test unreachable llvm.dbg.label
;
; RUN: llc -filetype=obj -split-dwarf-file debug.dwo -mtriple=x86_64-unknown-linux-gnu -o - %s | llvm-dwarfdump -v - | FileCheck %s
;
; CHECK: .debug_info.dwo contents:
; CHECK: DW_TAG_label
; CHECK-NEXT: DW_AT_name {{.*}}"done"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_AT_low_pc
; CHECK: DW_TAG_label
; CHECK-NEXT: DW_AT_name {{.*}}"removed"
; CHECK-NOT: DW_AT_low_pc
source_filename = "debug-label-unreached.c"

define dso_local i32 @foo(i32 %a, i32 %b) !dbg !8 {
entry:
  %sum = add nsw i32 %a, %b, !dbg !12
  call void @llvm.dbg.label(metadata !11), !dbg !12
  ret i32 %sum, !dbg !13
}

declare void @llvm.dbg.label(metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "debug-label-unreached.c", directory: "./")
!2 = !{}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!9 = !{!10, !11}
!10 = !DILabel(scope: !8, name: "removed", file: !1, line: 11)
!11 = !DILabel(scope: !8, name: "done", file: !1, line: 13)
!12 = !DILocation(line: 13, column: 1, scope: !8)
!13 = !DILocation(line: 14, column: 5, scope: !8)
