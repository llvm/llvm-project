;; This test checks the validity of DWARF operators DW_OP_rot, DW_OP_neg, and DW_OP_abs.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: !DIExpression(DW_OP_push_object_address, DW_OP_lit0, DW_OP_lit0, DW_OP_neg, DW_OP_abs, DW_OP_rot, DW_OP_rot, DW_OP_rot, DW_OP_plus, DW_OP_plus)

; ModuleID = 'DW_OP_rot_neg_abs.adb'
source_filename = "/dir/DW_OP_rot_neg_abs.ll"

!named = !{!DIExpression(DW_OP_push_object_address, DW_OP_lit0, DW_OP_lit0, DW_OP_neg, DW_OP_abs, DW_OP_rot, DW_OP_rot, DW_OP_rot, DW_OP_plus, DW_OP_plus)}
