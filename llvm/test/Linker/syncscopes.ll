; RUN: llvm-link %S/Inputs/syncscope-1.ll %S/Inputs/syncscope-2.ll -S | FileCheck %s

; CHECK-LABEL: define void @syncscope_1
; CHECK: fence syncscope("agent") seq_cst
; CHECK: fence syncscope("workgroup") seq_cst
; CHECK: fence syncscope("wavefront") seq_cst

; CHECK-LABEL: define void @syncscope_2
; CHECK: fence syncscope("image") seq_cst
; CHECK: fence syncscope("agent") seq_cst
; CHECK: fence syncscope("workgroup") seq_cst
