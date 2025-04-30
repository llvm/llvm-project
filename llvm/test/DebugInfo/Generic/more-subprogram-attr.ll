; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Make sure we're emitting DW_AT_{pure,elemental,recursive}.
; CHECK: DW_TAG_subprogram
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_AT_name {{.*}} "main"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_AT_pure [DW_FORM_flag_present] (true)
; CHECK:   DW_AT_elemental [DW_FORM_flag_present] (true)
; CHECK:   DW_AT_recursive [DW_FORM_flag_present] (true)

define dso_local i32 @main() !dbg !7 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "x.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "main", scope: !8, file: !8, line: 1, type: !9, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagPure | DISPFlagElemental | DISPFlagRecursive, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "x.c", directory: "/tmp")
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 3, column: 3, scope: !7)
