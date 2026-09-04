; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

; Labels emit no expression bytes, so they do not turn a simple register
; location into a memory location or interfere with fragment lookahead.

declare void @clobber()

define void @registers(i64 %x, i32 %simple_subreg, i32 %complex_subreg) !dbg !5 {
entry:
  #dbg_value(i64 %x, !9, !DIExpression(DW_OP_LLVM_label, 1), !12)
  #dbg_value(i32 %simple_subreg, !15,
             !DIExpression(DW_OP_LLVM_label, 3,
                           DW_OP_LLVM_fragment, 0, 32), !12)
  #dbg_value(i32 %complex_subreg, !16,
             !DIExpression(DW_OP_LLVM_label, 4,
                           DW_OP_plus_uconst, 1), !12)
  call void @clobber(), !dbg !12
  ret void, !dbg !12
}

; CHECK: DW_OP_reg5 RDI)
; CHECK: DW_AT_name {{.*}}"label_only"
; CHECK: DW_OP_reg4 RSI, DW_OP_piece 0x4)
; CHECK: DW_AT_name {{.*}}"simple_subreg"
; CHECK: DW_OP_breg1 RDX+0, DW_OP_constu 0xffffffff, DW_OP_and, DW_OP_plus_uconst 0x1)
; CHECK: DW_AT_name {{.*}}"complex_subreg"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "registers", scope: !1, file: !1,
                            type: !6, spFlags: DISPFlagDefinition, unit: !0,
                            retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9, !15, !16}
!9 = !DILocalVariable(name: "label_only", scope: !5, type: !11)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!12 = !DILocation(line: 1, scope: !5)
!15 = !DILocalVariable(name: "simple_subreg", scope: !5, type: !11)
!16 = !DILocalVariable(name: "complex_subreg", scope: !5, type: !17)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
