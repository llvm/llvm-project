; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

define void @test_cmpxchg(i32* %addr, i32 %desired, i32 %new) {
  cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  ; CHECK: cmpxchg ptr %addr, i32 %desired, i32 %new seq_cst seq_cst

  cmpxchg volatile i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  ; CHECK: cmpxchg volatile ptr %addr, i32 %desired, i32 %new seq_cst monotonic

  cmpxchg weak i32* %addr, i32 %desired, i32 %new acq_rel acquire
  ; CHECK: cmpxchg weak ptr %addr, i32 %desired, i32 %new acq_rel acquire

  cmpxchg weak volatile i32* %addr, i32 %desired, i32 %new syncscope("singlethread") release monotonic
  ; CHECK: cmpxchg weak volatile ptr %addr, i32 %desired, i32 %new syncscope("singlethread") release monotonic

  ret void
}

define float @test_atomicrmw_fmf(ptr %addr, float %value) {
  ; CHECK: %fast = atomicrmw fast fadd ptr %addr, float %value monotonic
  %fast = atomicrmw fast fadd ptr %addr, float %value monotonic

  ; CHECK: %flags = atomicrmw volatile nnan ninf fsub ptr %addr, float %value acquire, align 4
  %flags = atomicrmw volatile ninf nnan fsub ptr %addr, float %value acquire, align 4

  ; CHECK: %xchg = atomicrmw nsz xchg ptr %addr, float %value seq_cst
  %xchg = atomicrmw nsz xchg ptr %addr, float %value seq_cst
  ret float %xchg
}

define <2 x half> @test_atomicrmw_elementwise_fmf(ptr %addr,
                                                   <2 x half> %value) {
  ; CHECK: %old = atomicrmw volatile fast elementwise fadd ptr %addr, <2 x half> %value monotonic
  %old = atomicrmw volatile fast elementwise fadd ptr %addr, <2 x half> %value monotonic
  ret <2 x half> %old
}
