; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; Recursion: foo -> bar -> baz -> qux -> foo

; CHECK-LABEL: {{^}}qux
; CHECK: .set .Lqux.num_vgpr, max(71, .Lfoo.num_vgpr)
; CHECK: .set .Lqux.num_agpr, max(0, .Lfoo.num_agpr)
; CHECK: .set .Lqux.numbered_sgpr, max(46, .Lfoo.numbered_sgpr)
; CHECK: .set .Lqux.private_seg_size, 16
; CHECK: .set .Lqux.uses_vcc, or(1, .Lfoo.uses_vcc)
; CHECK: .set .Lqux.uses_flat_scratch, or(0, .Lfoo.uses_flat_scratch)
; CHECK: .set .Lqux.has_dyn_sized_stack, or(0, .Lfoo.has_dyn_sized_stack)
; CHECK: .set .Lqux.has_recursion, or(1, .Lfoo.has_recursion)
; CHECK: .set .Lqux.has_indirect_call, or(0, .Lfoo.has_indirect_call)

; CHECK-LABEL: {{^}}baz
; CHECK: .set .Lbaz.num_vgpr, max(61, .Lqux.num_vgpr)
; CHECK: .set .Lbaz.num_agpr, max(0, .Lqux.num_agpr)
; CHECK: .set .Lbaz.numbered_sgpr, max(51, .Lqux.numbered_sgpr)
; CHECK: .set .Lbaz.private_seg_size, 16+max(.Lqux.private_seg_size)
; CHECK: .set .Lbaz.uses_vcc, or(1, .Lqux.uses_vcc)
; CHECK: .set .Lbaz.uses_flat_scratch, or(0, .Lqux.uses_flat_scratch)
; CHECK: .set .Lbaz.has_dyn_sized_stack, or(0, .Lqux.has_dyn_sized_stack)
; CHECK: .set .Lbaz.has_recursion, or(1, .Lqux.has_recursion)
; CHECK: .set .Lbaz.has_indirect_call, or(0, .Lqux.has_indirect_call)

; CHECK-LABEL: {{^}}bar
; CHECK: .set .Lbar.num_vgpr, max(51, .Lbaz.num_vgpr)
; CHECK: .set .Lbar.num_agpr, max(0, .Lbaz.num_agpr)
; CHECK: .set .Lbar.numbered_sgpr, max(61, .Lbaz.numbered_sgpr)
; CHECK: .set .Lbar.private_seg_size, 16+max(.Lbaz.private_seg_size)
; CHECK: .set .Lbar.uses_vcc, or(1, .Lbaz.uses_vcc)
; CHECK: .set .Lbar.uses_flat_scratch, or(0, .Lbaz.uses_flat_scratch)
; CHECK: .set .Lbar.has_dyn_sized_stack, or(0, .Lbaz.has_dyn_sized_stack)
; CHECK: .set .Lbar.has_recursion, or(1, .Lbaz.has_recursion)
; CHECK: .set .Lbar.has_indirect_call, or(0, .Lbaz.has_indirect_call)

; CHECK-LABEL: {{^}}foo
; CHECK: .set .Lfoo.num_vgpr, max(46, 71)
; CHECK: .set .Lfoo.num_agpr, max(0, 0)
; CHECK: .set .Lfoo.numbered_sgpr, max(71, 61)
; CHECK: .set .Lfoo.private_seg_size, 16
; CHECK: .set .Lfoo.uses_vcc, 1
; CHECK: .set .Lfoo.uses_flat_scratch, 0
; CHECK: .set .Lfoo.has_dyn_sized_stack, 0
; CHECK: .set .Lfoo.has_recursion, 1
; CHECK: .set .Lfoo.has_indirect_call, 0

define void @foo() {
entry:
  call void @bar()
  call void asm sideeffect "", "~{v45}"()
  call void asm sideeffect "", "~{s70}"()
  ret void
}

define void @bar() {
entry:
  call void @baz()
  call void asm sideeffect "", "~{v50}"()
  call void asm sideeffect "", "~{s60}"()
  ret void
}

define void @baz() {
entry:
  call void @qux()
  call void asm sideeffect "", "~{v60}"()
  call void asm sideeffect "", "~{s50}"()
  ret void
}

define void @qux() {
entry:
  call void @foo()
  call void asm sideeffect "", "~{v70}"()
  call void asm sideeffect "", "~{s45}"()
  ret void
}

; CHECK-LABEL: {{^}}usefoo
; CHECK: .set .Lusefoo.num_vgpr, max(32, .Lfoo.num_vgpr)
; CHECK: .set .Lusefoo.num_agpr, max(0, .Lfoo.num_agpr)
; CHECK: .set .Lusefoo.numbered_sgpr, max(33, .Lfoo.numbered_sgpr)
; CHECK: .set .Lusefoo.private_seg_size, 0+max(.Lfoo.private_seg_size)
; CHECK: .set .Lusefoo.uses_vcc, or(1, .Lfoo.uses_vcc)
; CHECK: .set .Lusefoo.uses_flat_scratch, or(1, .Lfoo.uses_flat_scratch)
; CHECK: .set .Lusefoo.has_dyn_sized_stack, or(0, .Lfoo.has_dyn_sized_stack)
; CHECK: .set .Lusefoo.has_recursion, or(1, .Lfoo.has_recursion)
; CHECK: .set .Lusefoo.has_indirect_call, or(0, .Lfoo.has_indirect_call)
define amdgpu_kernel void @usefoo() {
  call void @foo()
  ret void
}

; Recursion: A -> B -> C -> A && C -> D -> C

; CHECK-LABEL: {{^}}D
; CHECK: .set .LD.num_vgpr, max(71, .LC.num_vgpr)
; CHECK: .set .LD.num_agpr, max(0, .LC.num_agpr)
; CHECK: .set .LD.numbered_sgpr, max(71, .LC.numbered_sgpr)
; CHECK: .set .LD.private_seg_size, 16+max(.LC.private_seg_size)
; CHECK: .set .LD.uses_vcc, or(1, .LC.uses_vcc)
; CHECK: .set .LD.uses_flat_scratch, or(0, .LC.uses_flat_scratch)
; CHECK: .set .LD.has_dyn_sized_stack, or(0, .LC.has_dyn_sized_stack)
; CHECK: .set .LD.has_recursion, or(1, .LC.has_recursion)
; CHECK: .set .LD.has_indirect_call, or(0, .LC.has_indirect_call)

; CHECK-LABEL: {{^}}C
; CHECK: .set .LC.num_vgpr, max(42, .LA.num_vgpr, 71)
; CHECK: .set .LC.num_agpr, max(0, .LA.num_agpr, 0)
; CHECK: .set .LC.numbered_sgpr, max(71, .LA.numbered_sgpr, 71)
; CHECK: .set .LC.private_seg_size, 16+max(.LA.private_seg_size)
; CHECK: .set .LC.uses_vcc, or(1, .LA.uses_vcc)
; CHECK: .set .LC.uses_flat_scratch, or(0, .LA.uses_flat_scratch)
; CHECK: .set .LC.has_dyn_sized_stack, or(0, .LA.has_dyn_sized_stack)
; CHECK: .set .LC.has_recursion, or(1, .LA.has_recursion)
; CHECK: .set .LC.has_indirect_call, or(0, .LA.has_indirect_call)

; CHECK-LABEL: {{^}}B
; CHECK: .set .LB.num_vgpr, max(42, .LC.num_vgpr)
; CHECK: .set .LB.num_agpr, max(0, .LC.num_agpr)
; CHECK: .set .LB.numbered_sgpr, max(71, .LC.numbered_sgpr)
; CHECK: .set .LB.private_seg_size, 16+max(.LC.private_seg_size)
; CHECK: .set .LB.uses_vcc, or(1, .LC.uses_vcc)
; CHECK: .set .LB.uses_flat_scratch, or(0, .LC.uses_flat_scratch)
; CHECK: .set .LB.has_dyn_sized_stack, or(0, .LC.has_dyn_sized_stack)
; CHECK: .set .LB.has_recursion, or(1, .LC.has_recursion)
; CHECK: .set .LB.has_indirect_call, or(0, .LC.has_indirect_call)

; CHECK-LABEL: {{^}}A
; CHECK: .set .LA.num_vgpr, max(42, 71)
; CHECK: .set .LA.num_agpr, max(0, 0)
; CHECK: .set .LA.numbered_sgpr, max(71, 71)
; CHECK: .set .LA.private_seg_size, 16
; CHECK: .set .LA.uses_vcc, 1
; CHECK: .set .LA.uses_flat_scratch, 0
; CHECK: .set .LA.has_dyn_sized_stack, 0
; CHECK: .set .LA.has_recursion, 1
; CHECK: .set .LA.has_indirect_call, 0

define void @A() {
  call void @B()
  call void asm sideeffect "", "~{v10}"()
  call void asm sideeffect "", "~{s50}"()
  ret void
}

define void @B() {
  call void @C()
  call void asm sideeffect "", "~{v20}"()
  call void asm sideeffect "", "~{s30}"()
  ret void
}

define void @C() {
  call void @A()
  call void @D()
  call void asm sideeffect "", "~{v30}"()
  call void asm sideeffect "", "~{s40}"()
  ret void
}

define void @D() {
  call void @C()
  call void asm sideeffect "", "~{v70}"()
  call void asm sideeffect "", "~{s70}"()
  ret void
}
