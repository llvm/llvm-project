; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - %s | llvm-dwarfdump -v - | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4 -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s --check-prefix=LEGACY

; Branch offsets come from the emitted bytes, so check:
;
; - zero, forward, backward, and cyclic branches;
; - ops which grow during lowering;
; - labels on either side of an emitted op; and
; - the fragment return path and DWARF 4 convert expansion.

define void @f() !dbg !5 {
entry:
  #dbg_value(i64 0, !9,
             !DIExpression(DW_OP_LLVM_bra, 1, DW_OP_LLVM_label, 1), !18)
  #dbg_value(i64 0, !10,
             !DIExpression(DW_OP_LLVM_label, 2, DW_OP_LLVM_skip, 2), !18)
  #dbg_value(i64 0, !11,
             !DIExpression(DW_OP_LLVM_label, 3, DW_OP_LLVM_bra, 4,
                           DW_OP_LLVM_skip, 3, DW_OP_LLVM_label, 4), !18)
  #dbg_value(i64 0, !12,
             !DIExpression(DW_OP_LLVM_bra, 5, DW_OP_deref_size, 1,
                           DW_OP_plus_uconst, 128,
                           DW_OP_LLVM_extract_bits_sext, 4, 4,
                           DW_OP_LLVM_label, 5), !18)
  #dbg_value(i64 0, !13,
             !DIExpression(DW_OP_LLVM_skip, 6,
                           DW_OP_LLVM_convert, 32, DW_ATE_signed,
                           DW_OP_LLVM_convert, 64, DW_ATE_signed,
                           DW_OP_LLVM_label, 6), !18)
  #dbg_value(i64 0, !14,
             !DIExpression(DW_OP_LLVM_bra, 7, DW_OP_LLVM_label, 7,
                           DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 32), !18)
  #dbg_value(i64 0, !15,
             !DIExpression(DW_OP_plus_uconst, 1, DW_OP_LLVM_label, 8,
                           DW_OP_LLVM_skip, 8), !18)
  #dbg_value(i64 0, !16,
             !DIExpression(DW_OP_LLVM_label, 9, DW_OP_plus_uconst, 1,
                           DW_OP_LLVM_skip, 9), !18)
  ret void, !dbg !18
}

; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_bra +0, DW_OP_stack_value)
; CHECK: DW_AT_name{{.*}}"zero"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_skip -3, DW_OP_stack_value)
; CHECK: DW_AT_name{{.*}}"backward"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_bra +3, DW_OP_skip -6, DW_OP_stack_value)
; CHECK: DW_AT_name{{.*}}"cycle"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_bra +11, DW_OP_deref_size 0x1, DW_OP_plus_uconst 0x80, DW_OP_constu 0x38, DW_OP_shl, DW_OP_constu 0x3c, DW_OP_shra, DW_OP_stack_value)
; CHECK: DW_AT_name{{.*}}"expanded"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_skip +10, DW_OP_convert {{.*}} "DW_ATE_signed_32", DW_OP_convert {{.*}} "DW_ATE_signed_64", DW_OP_stack_value)
; CHECK: DW_AT_name{{.*}}"convert"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_bra +0, DW_OP_stack_value, DW_OP_piece 0x4)
; CHECK: DW_AT_name{{.*}}"fragment"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_plus_uconst 0x1, DW_OP_skip -3, DW_OP_stack_value)
; CHECK: DW_AT_name{{.*}}"label_after_offset"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_lit0, DW_OP_plus_uconst 0x1, DW_OP_skip -5, DW_OP_stack_value)
; CHECK: DW_AT_name{{.*}}"label_before_offset"

; LEGACY: DW_AT_location (DW_OP_lit0, DW_OP_skip +11, DW_OP_dup, DW_OP_constu 0x1f, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x20, DW_OP_shl, DW_OP_or, DW_OP_stack_value)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !6,
                            spFlags: DISPFlagDefinition,
                            unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9, !10, !11, !12, !13, !14, !15, !16}
!9 = !DILocalVariable(name: "zero", scope: !5, type: !19)
!10 = !DILocalVariable(name: "backward", scope: !5, type: !19)
!11 = !DILocalVariable(name: "cycle", scope: !5, type: !19)
!12 = !DILocalVariable(name: "expanded", scope: !5, type: !19)
!13 = !DILocalVariable(name: "convert", scope: !5, type: !19)
!14 = !DILocalVariable(name: "fragment", scope: !5, type: !19)
!15 = !DILocalVariable(name: "label_after_offset", scope: !5, type: !19)
!16 = !DILocalVariable(name: "label_before_offset", scope: !5, type: !19)
!18 = !DILocation(line: 1, scope: !5)
!19 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
