; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 | FileCheck %s
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 | %ptxas-verify -arch=sm_90 %}
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx77 | FileCheck %s

declare i1 @llvm.nvvm.isspacep.grid_const(ptr) readnone noinline

; CHECK-LABEL: is_grid_const
define i1 @is_grid_const(ptr %addr) {
; CHECK: isspacep.param
  %v = tail call i1 @llvm.nvvm.isspacep.grid_const(ptr %addr)
  ret i1 %v
}
