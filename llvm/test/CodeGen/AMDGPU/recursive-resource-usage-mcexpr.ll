; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; Recursion: foo -> bar -> baz -> qux -> foo

; CHECK-LABEL: {{^}}qux
; CHECK: .set qux.num_vgpr, max(71, foo.num_vgpr)
; CHECK: .set qux.num_agpr, max(0, foo.num_agpr)
; CHECK: .set qux.numbered_sgpr, max(46, foo.numbered_sgpr)
; CHECK: .set qux.private_seg_size, 16
; CHECK: .set qux.uses_vcc, or(1, foo.uses_vcc)
; CHECK: .set qux.uses_flat_scratch, or(0, foo.uses_flat_scratch)
; CHECK: .set qux.has_dyn_sized_stack, or(0, foo.has_dyn_sized_stack)
; CHECK: .set qux.has_recursion, or(1, foo.has_recursion)
; CHECK: .set qux.has_indirect_call, or(0, foo.has_indirect_call)

; CHECK-LABEL: {{^}}baz
; CHECK: .set baz.num_vgpr, max(61, qux.num_vgpr)
; CHECK: .set baz.num_agpr, max(0, qux.num_agpr)
; CHECK: .set baz.numbered_sgpr, max(51, qux.numbered_sgpr)
; CHECK: .set baz.private_seg_size, 16+max(qux.private_seg_size)
; CHECK: .set baz.uses_vcc, or(1, qux.uses_vcc)
; CHECK: .set baz.uses_flat_scratch, or(0, qux.uses_flat_scratch)
; CHECK: .set baz.has_dyn_sized_stack, or(0, qux.has_dyn_sized_stack)
; CHECK: .set baz.has_recursion, or(1, qux.has_recursion)
; CHECK: .set baz.has_indirect_call, or(0, qux.has_indirect_call)

; CHECK-LABEL: {{^}}bar
; CHECK: .set bar.num_vgpr, max(51, baz.num_vgpr)
; CHECK: .set bar.num_agpr, max(0, baz.num_agpr)
; CHECK: .set bar.numbered_sgpr, max(61, baz.numbered_sgpr)
; CHECK: .set bar.private_seg_size, 16+max(baz.private_seg_size)
; CHECK: .set bar.uses_vcc, or(1, baz.uses_vcc)
; CHECK: .set bar.uses_flat_scratch, or(0, baz.uses_flat_scratch)
; CHECK: .set bar.has_dyn_sized_stack, or(0, baz.has_dyn_sized_stack)
; CHECK: .set bar.has_recursion, or(1, baz.has_recursion)
; CHECK: .set bar.has_indirect_call, or(0, baz.has_indirect_call)

; CHECK-LABEL: {{^}}foo
; CHECK: .set foo.num_vgpr, max(46, 71)
; CHECK: .set foo.num_agpr, max(0, 0)
; CHECK: .set foo.numbered_sgpr, max(71, 61)
; CHECK: .set foo.private_seg_size, 16
; CHECK: .set foo.uses_vcc, 1
; CHECK: .set foo.uses_flat_scratch, 0
; CHECK: .set foo.has_dyn_sized_stack, 0
; CHECK: .set foo.has_recursion, 1
; CHECK: .set foo.has_indirect_call, 0

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
; CHECK: .set usefoo.num_vgpr, max(32, foo.num_vgpr)
; CHECK: .set usefoo.num_agpr, max(0, foo.num_agpr)
; CHECK: .set usefoo.numbered_sgpr, max(33, foo.numbered_sgpr)
; CHECK: .set usefoo.private_seg_size, 0+max(foo.private_seg_size)
; CHECK: .set usefoo.uses_vcc, or(1, foo.uses_vcc)
; CHECK: .set usefoo.uses_flat_scratch, or(1, foo.uses_flat_scratch)
; CHECK: .set usefoo.has_dyn_sized_stack, or(0, foo.has_dyn_sized_stack)
; CHECK: .set usefoo.has_recursion, or(1, foo.has_recursion)
; CHECK: .set usefoo.has_indirect_call, or(0, foo.has_indirect_call)
define amdgpu_kernel void @usefoo() {
  call void @foo()
  ret void
}

; Recursion: A -> B -> C -> A && C -> D -> C

; CHECK-LABEL: {{^}}D
; CHECK: .set D.num_vgpr, max(71, C.num_vgpr)
; CHECK: .set D.num_agpr, max(0, C.num_agpr)
; CHECK: .set D.numbered_sgpr, max(71, C.numbered_sgpr)
; CHECK: .set D.private_seg_size, 16+max(C.private_seg_size)
; CHECK: .set D.uses_vcc, or(1, C.uses_vcc)
; CHECK: .set D.uses_flat_scratch, or(0, C.uses_flat_scratch)
; CHECK: .set D.has_dyn_sized_stack, or(0, C.has_dyn_sized_stack)
; CHECK: .set D.has_recursion, or(1, C.has_recursion)
; CHECK: .set D.has_indirect_call, or(0, C.has_indirect_call)

; CHECK-LABEL: {{^}}C
; CHECK: .set C.num_vgpr, max(42, A.num_vgpr, 71)
; CHECK: .set C.num_agpr, max(0, A.num_agpr, 0)
; CHECK: .set C.numbered_sgpr, max(71, A.numbered_sgpr, 71)
; CHECK: .set C.private_seg_size, 16+max(A.private_seg_size)
; CHECK: .set C.uses_vcc, or(1, A.uses_vcc)
; CHECK: .set C.uses_flat_scratch, or(0, A.uses_flat_scratch)
; CHECK: .set C.has_dyn_sized_stack, or(0, A.has_dyn_sized_stack)
; CHECK: .set C.has_recursion, or(1, A.has_recursion)
; CHECK: .set C.has_indirect_call, or(0, A.has_indirect_call)

; CHECK-LABEL: {{^}}B
; CHECK: .set B.num_vgpr, max(42, C.num_vgpr)
; CHECK: .set B.num_agpr, max(0, C.num_agpr)
; CHECK: .set B.numbered_sgpr, max(71, C.numbered_sgpr)
; CHECK: .set B.private_seg_size, 16+max(C.private_seg_size)
; CHECK: .set B.uses_vcc, or(1, C.uses_vcc)
; CHECK: .set B.uses_flat_scratch, or(0, C.uses_flat_scratch)
; CHECK: .set B.has_dyn_sized_stack, or(0, C.has_dyn_sized_stack)
; CHECK: .set B.has_recursion, or(1, C.has_recursion)
; CHECK: .set B.has_indirect_call, or(0, C.has_indirect_call)

; CHECK-LABEL: {{^}}A
; CHECK: .set A.num_vgpr, max(42, 71)
; CHECK: .set A.num_agpr, max(0, 0)
; CHECK: .set A.numbered_sgpr, max(71, 71)
; CHECK: .set A.private_seg_size, 16
; CHECK: .set A.uses_vcc, 1
; CHECK: .set A.uses_flat_scratch, 0
; CHECK: .set A.has_dyn_sized_stack, 0
; CHECK: .set A.has_recursion, 1
; CHECK: .set A.has_indirect_call, 0

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
