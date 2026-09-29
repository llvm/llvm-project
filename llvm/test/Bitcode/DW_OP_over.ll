;; This test checks processing of DWARF operator DW_OP_over.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 88, DW_OP_plus, DW_OP_deref)

; ModuleID = 'DW_OP_over.f90'
source_filename = "/dir/DW_OP_over.ll"

!named = !{!DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 88, DW_OP_plus, DW_OP_deref)}
