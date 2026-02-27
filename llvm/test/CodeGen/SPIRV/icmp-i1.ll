; RUN: llc -O0 -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj < %s  | spirv-val %}

define spir_func i1 @ugt(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function ugt
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp ugt i1 %p, %q
  ret i1 %r
}

define spir_func i1 @ugt_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function ugt_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp samesign ugt i1 %p, %q
  ret i1 %r
}

define spir_func i1 @uge(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function uge
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp uge i1 %p, %q
  ret i1 %r
}

define spir_func i1 @uge_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function uge_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp samesign uge i1 %p, %q
  ret i1 %r
}

define spir_func i1 @ult(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function ult
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp ult i1 %p, %q
  ret i1 %r
}

define spir_func i1 @ult_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function ult_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp samesign ult i1 %p, %q
  ret i1 %r
}

define spir_func i1 @ule(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function ule
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp ule i1 %p, %q
  ret i1 %r
}

define spir_func i1 @ule_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function ule_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp samesign ule i1 %p, %q
  ret i1 %r
}

define spir_func i1 @sgt(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function sgt
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp sgt i1 %p, %q
  ret i1 %r
}

define spir_func i1 @sgt_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function sgt_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp samesign sgt i1 %p, %q
  ret i1 %r
}

define spir_func i1 @sge(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function sge
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp sge i1 %p, %q
  ret i1 %r
}

define spir_func i1 @sge_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function sge_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp samesign sge i1 %p, %q
  ret i1 %r
}

define spir_func i1 @slt(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function slt
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp slt i1 %p, %q
  ret i1 %r
}

define spir_func i1 @slt_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function slt_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalAnd
  %r = icmp samesign slt i1 %p, %q
  ret i1 %r
}

define spir_func i1 @sle(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function sle
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp sle i1 %p, %q
  ret i1 %r
}

define spir_func i1 @sle_same_sign(i1 %p, i1 %q) addrspace(4) {
  ; CHECK: Begin function sle_same_sign
  ; CHECK: OpLogicalNotEqual
  ; CHECK: OpLogicalOr
  %r = icmp samesign sle i1 %p, %q
  ret i1 %r
}
