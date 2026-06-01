; RUN: llc < %s -mtriple=nvptx -mcpu=sm_60 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas-sm_60 && ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}
; RUN: %if ptxas-sm_60 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}

; CHECK-LABEL: .func test_scoped_atomicrmw(
define void @test_scoped_atomicrmw(ptr %p, i32 %i, i64 %ll, float %f, double %d) {
  ; block (cta) scope
; CHECK: atom.cta.add.u32
  %01 = atomicrmw add  ptr %p, i32 %i   syncscope("block") monotonic
; CHECK: atom.cta.add.u64
  %02 = atomicrmw add  ptr %p, i64 %ll  syncscope("block") monotonic
; CHECK: atom.cta.add.f32
  %03 = atomicrmw fadd ptr %p, float %f syncscope("block") monotonic
; CHECK: atom.cta.add.f64
  %04 = atomicrmw fadd ptr %p, double %d syncscope("block") monotonic
; CHECK: atom.cta.exch.b32
  %05 = atomicrmw xchg ptr %p, i32 %i   syncscope("block") monotonic
; CHECK: atom.cta.max.s32
  %06 = atomicrmw max  ptr %p, i32 %i   syncscope("block") monotonic
; CHECK: atom.cta.max.u32
  %07 = atomicrmw umax ptr %p, i32 %i   syncscope("block") monotonic
; CHECK: atom.cta.min.s64
  %08 = atomicrmw min  ptr %p, i64 %ll  syncscope("block") monotonic
; CHECK: atom.cta.min.u64
  %09 = atomicrmw umin ptr %p, i64 %ll  syncscope("block") monotonic
; CHECK: atom.cta.and.b32
  %10 = atomicrmw and  ptr %p, i32 %i   syncscope("block") monotonic
; CHECK: atom.cta.or.b32
  %11 = atomicrmw or   ptr %p, i32 %i   syncscope("block") monotonic
; CHECK: atom.cta.xor.b32
  %12 = atomicrmw xor  ptr %p, i32 %i   syncscope("block") monotonic
; CHECK: atom.cta.inc.u32
  %13 = atomicrmw uinc_wrap ptr %p, i32 %i syncscope("block") monotonic
; CHECK: atom.cta.dec.u32
  %14 = atomicrmw udec_wrap ptr %p, i32 %i syncscope("block") monotonic
; CHECK: atom.cta.cas.b32
  %15 = cmpxchg ptr %p, i32 %i, i32 %i syncscope("block") monotonic monotonic

  ; system scope (default)
; CHECK: atom.sys.add.u32
  %21 = atomicrmw add  ptr %p, i32 %i   monotonic
; CHECK: atom.sys.max.u32
  %22 = atomicrmw umax ptr %p, i32 %i   monotonic
; CHECK: atom.sys.min.s32
  %23 = atomicrmw min  ptr %p, i32 %i   monotonic
; CHECK: atom.sys.cas.b64
  %24 = cmpxchg ptr %p, i64 %ll, i64 %ll monotonic monotonic
  ret void
}
