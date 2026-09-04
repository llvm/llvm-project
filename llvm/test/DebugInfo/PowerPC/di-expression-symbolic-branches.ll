; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=asm -o - %s | FileCheck %s

; We patch branches in two temporary buffers, so make sure both use the PowerPC
; byte order: a forward skip in a location list and a backward branch in an
; inline expression.

define void @f(i64 %x) !dbg !5 {
entry:
  #dbg_value(i64 0, !9,
             !DIExpression(DW_OP_LLVM_label, 2, DW_OP_LLVM_bra, 2), !11)
  #dbg_value(i64 %x, !10,
             !DIExpression(DW_OP_LLVM_skip, 3, DW_OP_plus_uconst, 1,
                           DW_OP_LLVM_label, 3), !11)
  call void @clobber(), !dbg !11
  ret void, !dbg !11
}

declare void @clobber()

; CHECK: .section .debug_loclists
; CHECK: .byte 47{{.*}}# DW_OP_skip
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .byte 2
; CHECK: .byte 5{{.*}}# DW_AT_location
; CHECK-NEXT: .byte 48
; CHECK-NEXT: .byte 40
; CHECK-NEXT: .short 65533

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, type: !6,
                            spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!9 = !DILocalVariable(name: "backward", scope: !5, type: !12)
!10 = !DILocalVariable(name: "value", arg: 1, scope: !5, type: !12)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
