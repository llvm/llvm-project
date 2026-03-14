; RUN: llc -mtriple armv7-linux -relocation-model=rwpi -o - %s | FileCheck %s
; RUN: llc -mtriple armv7-linux -relocation-model=ropi-rwpi -o - %s | FileCheck %s

@global = global i32 5, align 4, !dbg !0

; 8 bytes of data
; CHECK:      .byte 8 @ DW_AT_location
; DW_OP_const4u
; CHECK-NEXT: .byte 12
; relocation offset
; CHECK-NEXT: .long global(sbrel)
; DW_OP_breg9
; CHECK-NEXT: .byte 121
; offset from breg9
; CHECK-NEXT: .byte 0
; DW_OP_plus
; CHECK-NEXT: .byte 34

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 9f5c70c7ad404f0cb52416a0574d9e48d520be5d)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "rwpi.c", directory: "/tmp")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 1, !"min_enum_size", i32 4}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 9f5c70c7ad404f0cb52416a0574d9e48d520be5d)"}
