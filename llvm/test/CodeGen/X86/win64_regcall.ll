; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s

define dso_local x86_regcallcc void @ensure_align() local_unnamed_addr #0 {
entry:
  %b = alloca i32, align 4
  call void asm sideeffect "nopl $0", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) nonnull %b)
  ret void
}

; CHECK-LABEL: ensure_align: # @ensure_align
; CHECK: .seh_stackalloc 8
; CHECK: .seh_endprologue
