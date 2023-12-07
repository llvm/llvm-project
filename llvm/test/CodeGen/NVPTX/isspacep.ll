; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

declare i1 @llvm.nvvm.isspacep.const(ptr) readnone noinline
declare i1 @llvm.nvvm.isspacep.global(ptr) readnone noinline
declare i1 @llvm.nvvm.isspacep.local(ptr) readnone noinline
declare i1 @llvm.nvvm.isspacep.shared(ptr) readnone noinline

; CHECK: is_const
define i1 @is_const(ptr %addr) {
; CHECK: isspacep.const
  %v = tail call i1 @llvm.nvvm.isspacep.const(ptr %addr)
  ret i1 %v
}

; CHECK: is_global
define i1 @is_global(ptr %addr) {
; CHECK: isspacep.global
  %v = tail call i1 @llvm.nvvm.isspacep.global(ptr %addr)
  ret i1 %v
}

; CHECK: is_local
define i1 @is_local(ptr %addr) {
; CHECK: isspacep.local
  %v = tail call i1 @llvm.nvvm.isspacep.local(ptr %addr)
  ret i1 %v
}

; CHECK: is_shared
define i1 @is_shared(ptr %addr) {
; CHECK: isspacep.shared
  %v = tail call i1 @llvm.nvvm.isspacep.shared(ptr %addr)
  ret i1 %v
}

