; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

; Tag offsets emit no expression bytes, so skip them when deciding whether a
; register expression is complex.

define void @registers(i64 %tag_only, i32 %simple_subreg,
                       i32 %complex_subreg) !dbg !5 {
  #dbg_value(i64 %tag_only, !9,
             !DIExpression(DW_OP_LLVM_tag_offset, 7), !12)
  #dbg_value(i32 %simple_subreg, !10,
             !DIExpression(DW_OP_LLVM_tag_offset, 8,
                           DW_OP_LLVM_fragment, 0, 32), !12)
  #dbg_value(i32 %complex_subreg, !14,
             !DIExpression(DW_OP_LLVM_tag_offset, 9,
                           DW_OP_plus_uconst, 1), !12)
  ret void, !dbg !12
}

; A tag-only expression stays in a register.
; CHECK: DW_OP_reg5 RDI)
; CHECK: DW_AT_LLVM_tag_offset (0x07)
; CHECK: DW_AT_name ("tag_only")

; A tag before a fragment keeps the subregister location simple.
; CHECK: DW_OP_reg4 RSI, DW_OP_piece 0x4)
; CHECK: DW_AT_LLVM_tag_offset (0x08)
; CHECK: DW_AT_name ("simple_subreg")

; A real operation after the tag uses the complex register path and masks the
; 32-bit value before applying the operation.
; CHECK: DW_OP_breg1 RDX+0, DW_OP_constu 0xffffffff, DW_OP_and, DW_OP_plus_uconst 0x1)
; CHECK: DW_AT_LLVM_tag_offset (0x09)
; CHECK: DW_AT_name ("complex_subreg")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "registers", scope: !1, file: !1,
                            type: !6, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!9 = !DILocalVariable(name: "tag_only", scope: !5, type: !11)
!10 = !DILocalVariable(name: "simple_subreg", scope: !5, type: !11)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!12 = !DILocation(line: 1, scope: !5)
!14 = !DILocalVariable(name: "complex_subreg", scope: !5, type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
