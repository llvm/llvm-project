; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; CHECK-LABEL: {{^}}qux
; CHECK: .set qux.num_vgpr, max(41, foo.num_vgpr)
; CHECK: .set qux.num_agpr, max(0, foo.num_agpr)
; CHECK: .set qux.numbered_sgpr, max(34, foo.numbered_sgpr)
; CHECK: .set qux.private_seg_size, 16
; CHECK: .set qux.uses_vcc, or(1, foo.uses_vcc)
; CHECK: .set qux.uses_flat_scratch, or(0, foo.uses_flat_scratch)
; CHECK: .set qux.has_dyn_sized_stack, or(0, foo.has_dyn_sized_stack)
; CHECK: .set qux.has_recursion, or(1, foo.has_recursion)
; CHECK: .set qux.has_indirect_call, or(0, foo.has_indirect_call)

; CHECK-LABEL: {{^}}baz
; CHECK: .set baz.num_vgpr, max(42, qux.num_vgpr)
; CHECK: .set baz.num_agpr, max(0, qux.num_agpr)
; CHECK: .set baz.numbered_sgpr, max(34, qux.numbered_sgpr)
; CHECK: .set baz.private_seg_size, 16+(max(qux.private_seg_size))
; CHECK: .set baz.uses_vcc, or(1, qux.uses_vcc)
; CHECK: .set baz.uses_flat_scratch, or(0, qux.uses_flat_scratch)
; CHECK: .set baz.has_dyn_sized_stack, or(0, qux.has_dyn_sized_stack)
; CHECK: .set baz.has_recursion, or(1, qux.has_recursion)
; CHECK: .set baz.has_indirect_call, or(0, qux.has_indirect_call)

; CHECK-LABEL: {{^}}bar
; CHECK: .set bar.num_vgpr, max(42, baz.num_vgpr)
; CHECK: .set bar.num_agpr, max(0, baz.num_agpr)
; CHECK: .set bar.numbered_sgpr, max(34, baz.numbered_sgpr)
; CHECK: .set bar.private_seg_size, 16+(max(baz.private_seg_size))
; CHECK: .set bar.uses_vcc, or(1, baz.uses_vcc)
; CHECK: .set bar.uses_flat_scratch, or(0, baz.uses_flat_scratch)
; CHECK: .set bar.has_dyn_sized_stack, or(0, baz.has_dyn_sized_stack)
; CHECK: .set bar.has_recursion, or(1, baz.has_recursion)
; CHECK: .set bar.has_indirect_call, or(0, baz.has_indirect_call)

; CHECK-LABEL: {{^}}foo
; CHECK: .set foo.num_vgpr, 42
; CHECK: .set foo.num_agpr, 0
; CHECK: .set foo.numbered_sgpr, 34
; CHECK: .set foo.private_seg_size, 16
; CHECK: .set foo.uses_vcc, 1
; CHECK: .set foo.uses_flat_scratch, 0
; CHECK: .set foo.has_dyn_sized_stack, 0
; CHECK: .set foo.has_recursion, 1
; CHECK: .set foo.has_indirect_call, 0

define void @foo() {
entry:
  call void @bar()
  ret void
}

define void @bar() {
entry:
  call void @baz()
  ret void
}

define void @baz() {
entry:
  call void @qux()
  ret void
}

define void @qux() {
entry:
  call void @foo()
  ret void
}

; CHECK-LABEL: {{^}}usefoo
; CHECK: .set usefoo.num_vgpr, max(32, foo.num_vgpr)
; CHECK: .set usefoo.num_agpr, max(0, foo.num_agpr)
; CHECK: .set usefoo.numbered_sgpr, max(33, foo.numbered_sgpr)
; CHECK: .set usefoo.private_seg_size, 0+(max(foo.private_seg_size))
; CHECK: .set usefoo.uses_vcc, or(1, foo.uses_vcc)
; CHECK: .set usefoo.uses_flat_scratch, or(1, foo.uses_flat_scratch)
; CHECK: .set usefoo.has_dyn_sized_stack, or(0, foo.has_dyn_sized_stack)
; CHECK: .set usefoo.has_recursion, or(1, foo.has_recursion)
; CHECK: .set usefoo.has_indirect_call, or(0, foo.has_indirect_call)
define amdgpu_kernel void @usefoo() {
  call void @foo()
  ret void
}

