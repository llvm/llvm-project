; This checks that the list of index entries in the name index is terminated
; with a 1-byte value.

; RUN: llc -mtriple x86_64 -accel-tables=Dwarf -dwarf-version=5 -filetype=asm %s -o - | \
; RUN:   FileCheck %s

; CHECK:   .section .debug_names,"",@progbits
; CHECK: .Lnames_entries0:
; CHECK:   .byte 0
; CHECK-NEXT:   # End of list: int
; CHECK:   .byte 0
; CHECK-NEXT:   # End of list: foo

@foo = common dso_local global i32 0, align 4, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Manual", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!3 = !{}
!4 = !{!5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
