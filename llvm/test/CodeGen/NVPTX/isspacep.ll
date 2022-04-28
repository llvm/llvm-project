; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

declare i1 @llvm.nvvm.isspacep.const(i8*) readnone noinline
declare i1 @llvm.nvvm.isspacep.global(i8*) readnone noinline
declare i1 @llvm.nvvm.isspacep.local(i8*) readnone noinline
declare i1 @llvm.nvvm.isspacep.shared(i8*) readnone noinline

; CHECK: is_const
define i1 @is_const(i8* %addr) {
; CHECK: isspacep.const
  %v = tail call i1 @llvm.nvvm.isspacep.const(i8* %addr)
  ret i1 %v
}

; CHECK: is_global
define i1 @is_global(i8* %addr) {
; CHECK: isspacep.global
  %v = tail call i1 @llvm.nvvm.isspacep.global(i8* %addr)
  ret i1 %v
}

; CHECK: is_local
define i1 @is_local(i8* %addr) {
; CHECK: isspacep.local
  %v = tail call i1 @llvm.nvvm.isspacep.local(i8* %addr)
  ret i1 %v
}

; CHECK: is_shared
define i1 @is_shared(i8* %addr) {
; CHECK: isspacep.shared
  %v = tail call i1 @llvm.nvvm.isspacep.shared(i8* %addr)
  ret i1 %v
}

