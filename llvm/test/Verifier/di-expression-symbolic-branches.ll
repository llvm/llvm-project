; RUN: split-file %s %t
; RUN: opt -passes=verify -disable-output %t/valid.ll
; RUN: not opt -passes=verify -disable-output %t/arity.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/duplicate.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/missing.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/raw.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/raw-after-register.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/incompatible.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/location-arg.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/tag-ordering.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"
; RUN: not opt -passes=verify -disable-output %t/terminal.ll 2>&1 | FileCheck %s --check-prefix=INVALID --implicit-check-not="invalid expression"

; Check the verifier rules separately:
;
; - a label needs an ID;
; - labels are unique and each branch target exists;
; - raw branches and incompatible ops are rejected; and
; - tag_offset stays before control flow, which stays before stack_value.

; DIArgList doesn't support symbolic branches, but normal IR loading drops the
; bad debug info, so opt still succeeds.
; RUN: opt -passes=verify -disable-output %t/arg-list.ll 2>&1 | FileCheck %s --check-prefix=ARG-LIST

; INVALID: invalid expression
; ARG-LIST: DIArgList doesn't support symbolic branches
; ARG-LIST: warning: ignoring invalid debug info

;--- valid.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_bra, 0, DW_OP_LLVM_skip, 0,
                   DW_OP_LLVM_label, 0)

;--- arity.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_label)

;--- duplicate.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_label, 1, DW_OP_LLVM_label, 1)

;--- missing.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_bra, 1)

;--- raw.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_bra, 0)

;--- raw-after-register.ll
; A register normally ends validation, but it must not hide a raw branch later
; in the expression.
!named = !{!0}
!0 = !DIExpression(DW_OP_reg0, DW_OP_skip, 0)

;--- incompatible.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_implicit_pointer, DW_OP_LLVM_label, 1)

;--- location-arg.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_skip, 1,
                   DW_OP_LLVM_label, 1)

;--- tag-ordering.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_label, 1, DW_OP_LLVM_tag_offset, 0)

;--- terminal.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_stack_value, DW_OP_LLVM_label, 1)

;--- arg-list.ll
; The expression is valid by itself, so this diagnostic comes from DIArgList.
define void @f(i32 %x) !dbg !4 {
entry:
  #dbg_value(!DIArgList(i32 %x), !7,
             !DIExpression(DW_OP_LLVM_skip, 1, DW_OP_LLVM_label, 1), !8)
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, type: !5,
                            spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocalVariable(name: "x", scope: !4)
!8 = !DILocation(line: 1, column: 1, scope: !4)
