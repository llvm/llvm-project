; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -mcpu=gfx90a < %s | FileCheck %s

; CHECK-LABEL: {{^}}qux
; CHECK: .set qux.num_vgpr, 41
; CHECK: .set qux.num_agpr, 0
; CHECK: .set qux.numbered_sgpr, 34
; CHECK: .set qux.private_seg_size, 16
; CHECK: .set qux.uses_vcc, 1
; CHECK: .set qux.uses_flat_scratch, 0
; CHECK: .set qux.has_dyn_sized_stack, 0
; CHECK: .set qux.has_recursion, 1
; CHECK: .set qux.has_indirect_call, 0

; CHECK-LABEL: {{^}}baz
; CHECK: .set baz.num_vgpr, 42
; CHECK: .set baz.num_agpr, 0
; CHECK: .set baz.numbered_sgpr, 34
; CHECK: .set baz.private_seg_size, 16
; CHECK: .set baz.uses_vcc, 1
; CHECK: .set baz.uses_flat_scratch, 0
; CHECK: .set baz.has_dyn_sized_stack, 0
; CHECK: .set baz.has_recursion, 1
; CHECK: .set baz.has_indirect_call, 0

; CHECK-LABEL: {{^}}bar
; CHECK: .set bar.num_vgpr, 42
; CHECK: .set bar.num_agpr, 0
; CHECK: .set bar.numbered_sgpr, 34
; CHECK: .set bar.private_seg_size, 16
; CHECK: .set bar.uses_vcc, 1
; CHECK: .set bar.uses_flat_scratch, 0
; CHECK: .set bar.has_dyn_sized_stack, 0
; CHECK: .set bar.has_recursion, 1
; CHECK: .set bar.has_indirect_call, 0

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
; CHECK: .set usefoo.num_vgpr, 32
; CHECK: .set usefoo.num_agpr, 0
; CHECK: .set usefoo.numbered_sgpr, 33
; CHECK: .set usefoo.private_seg_size, 0
; CHECK: .set usefoo.uses_vcc, 1
; CHECK: .set usefoo.uses_flat_scratch, 1
; CHECK: .set usefoo.has_dyn_sized_stack, 0
; CHECK: .set usefoo.has_recursion, 1
; CHECK: .set usefoo.has_indirect_call, 0
define amdgpu_kernel void @usefoo() {
  call void @foo()
  ret void
}

