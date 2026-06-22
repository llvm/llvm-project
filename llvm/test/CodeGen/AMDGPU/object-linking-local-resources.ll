; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=asm < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=asm < %s | FileCheck %s --check-prefix=OL

declare void @extern_callee()

define void @calls_extern() {
  call void @extern_callee()
  ret void
}

define void @calls_indirect(ptr %fptr) {
  call void %fptr()
  ret void
}

define void @calls_local() {
  ret void
}

define amdgpu_kernel void @my_kernel(ptr %fptr) {
  call void @calls_extern()
  call void @calls_indirect(ptr %fptr)
  call void @calls_local()
  ret void
}

; COM: Default mode: direct-to-extern triggers the conservative "unknown
; COM: callee" path. Register/stack-size symbols include the module-level
; COM: sinks; boolean flags are all forced to 1; HasIndirectCall is set too
; COM: (IsIndirect covers calls to declarations).
; DEFAULT:       .set .Lcalls_extern.num_vgpr, max({{[0-9]+}}, amdgpu.max_num_vgpr)
; DEFAULT:       .set .Lcalls_extern.num_agpr, max({{[0-9]+}}, amdgpu.max_num_agpr)
; DEFAULT:       .set .Lcalls_extern.numbered_sgpr, max({{[0-9]+}}, amdgpu.max_num_sgpr)
; DEFAULT:       .set .Lcalls_extern.num_named_barrier, max({{[0-9]+}}, amdgpu.max_num_named_barrier)
; DEFAULT:       .set .Lcalls_extern.uses_vcc, 1
; DEFAULT:       .set .Lcalls_extern.uses_flat_scratch, 1
; DEFAULT:       .set .Lcalls_extern.has_dyn_sized_stack, 1
; DEFAULT:       .set .Lcalls_extern.has_recursion, 1
; DEFAULT:       .set .Lcalls_extern.has_indirect_call, 1

; COM: Object linking: the same function reports only its own local usage.
; COM: The sinks drop out of the register/stack-size expressions and the
; COM: pessimized boolean flags collapse to the true local values (UsesVCC is
; COM: still 1 here because the call-site lowering on gfx900 genuinely uses
; COM: VCC).
; OL:            .set .Lcalls_extern.num_vgpr, {{[0-9]+}}
; OL:            .set .Lcalls_extern.num_agpr, {{[0-9]+}}
; OL:            .set .Lcalls_extern.numbered_sgpr, {{[0-9]+}}
; OL:            .set .Lcalls_extern.num_named_barrier, {{[0-9]+}}
; OL:            .set .Lcalls_extern.uses_vcc, 1
; OL:            .set .Lcalls_extern.uses_flat_scratch, 0
; OL:            .set .Lcalls_extern.has_dyn_sized_stack, 0
; OL:            .set .Lcalls_extern.has_recursion, 0
; OL:            .set .Lcalls_extern.has_indirect_call, 1

; COM: True indirect call: same DEFAULT-vs-OL behavior as the direct-to-extern
; COM: case above. In DEFAULT mode all the flags are pessimized; with object
; COM: linking only HasIndirectCall is preserved (the linker sees the call
; COM: site's typeid and address-taken set and handles propagation).
; DEFAULT:       .set .Lcalls_indirect.uses_vcc, 1
; DEFAULT:       .set .Lcalls_indirect.uses_flat_scratch, 1
; DEFAULT:       .set .Lcalls_indirect.has_dyn_sized_stack, 1
; DEFAULT:       .set .Lcalls_indirect.has_recursion, 1
; DEFAULT:       .set .Lcalls_indirect.has_indirect_call, 1

; OL:            .set .Lcalls_indirect.uses_vcc, 1
; OL:            .set .Lcalls_indirect.uses_flat_scratch, 0
; OL:            .set .Lcalls_indirect.has_dyn_sized_stack, 0
; OL:            .set .Lcalls_indirect.has_recursion, 0
; OL:            .set .Lcalls_indirect.has_indirect_call, 1

; COM: Baseline: a function that makes no calls outside itself reports the
; COM: same all-zero local flags in both modes.
; DEFAULT:       .set .Lcalls_local.uses_vcc, 0
; DEFAULT:       .set .Lcalls_local.uses_flat_scratch, 0
; DEFAULT:       .set .Lcalls_local.has_dyn_sized_stack, 0
; DEFAULT:       .set .Lcalls_local.has_recursion, 0
; DEFAULT:       .set .Lcalls_local.has_indirect_call, 0

; OL:            .set .Lcalls_local.uses_vcc, 0
; OL:            .set .Lcalls_local.uses_flat_scratch, 0
; OL:            .set .Lcalls_local.has_dyn_sized_stack, 0
; OL:            .set .Lcalls_local.has_recursion, 0
; OL:            .set .Lcalls_local.has_indirect_call, 0

; COM: Kernel side of the DEFAULT-vs-OL comparison. DEFAULT mode emits
; COM: call-graph-propagation expressions (max()/or() over every callee's
; COM: symbols) so the kernel picks up its callees' pessimized values; object
; COM: linking emits concrete literals and leaves cross-TU aggregation to the
; COM: linker.
; DEFAULT:       .set .Lmy_kernel.num_vgpr, max({{[0-9]+}}, .Lcalls_extern.num_vgpr, .Lcalls_indirect.num_vgpr, .Lcalls_local.num_vgpr)
; DEFAULT:       .set .Lmy_kernel.num_agpr, max({{[0-9]+}}, .Lcalls_extern.num_agpr, .Lcalls_indirect.num_agpr, .Lcalls_local.num_agpr)
; DEFAULT:       .set .Lmy_kernel.num_named_barrier, max({{[0-9]+}}, .Lcalls_extern.num_named_barrier, .Lcalls_indirect.num_named_barrier, .Lcalls_local.num_named_barrier)
; DEFAULT:       .set .Lmy_kernel.private_seg_size, {{[0-9]+}}+max(.Lcalls_extern.private_seg_size, .Lcalls_indirect.private_seg_size, .Lcalls_local.private_seg_size)
; DEFAULT:       .set .Lmy_kernel.uses_vcc, or({{[0-9]+}}, .Lcalls_extern.uses_vcc, .Lcalls_indirect.uses_vcc, .Lcalls_local.uses_vcc)
; DEFAULT:       .set .Lmy_kernel.uses_flat_scratch, or({{[0-9]+}}, .Lcalls_extern.uses_flat_scratch, .Lcalls_indirect.uses_flat_scratch, .Lcalls_local.uses_flat_scratch)
; DEFAULT:       .set .Lmy_kernel.has_dyn_sized_stack, or({{[0-9]+}}, .Lcalls_extern.has_dyn_sized_stack, .Lcalls_indirect.has_dyn_sized_stack, .Lcalls_local.has_dyn_sized_stack)
; DEFAULT:       .set .Lmy_kernel.has_recursion, or({{[0-9]+}}, .Lcalls_extern.has_recursion, .Lcalls_indirect.has_recursion, .Lcalls_local.has_recursion)
; DEFAULT:       .set .Lmy_kernel.has_indirect_call, or({{[0-9]+}}, .Lcalls_extern.has_indirect_call, .Lcalls_indirect.has_indirect_call, .Lcalls_local.has_indirect_call)

; OL:            .set .Lmy_kernel.num_vgpr, {{[0-9]+}}
; OL:            .set .Lmy_kernel.num_agpr, {{[0-9]+}}
; OL:            .set .Lmy_kernel.num_named_barrier, {{[0-9]+}}
; OL:            .set .Lmy_kernel.private_seg_size, {{[0-9]+}}
; OL:            .set .Lmy_kernel.uses_vcc, {{[01]}}
; OL:            .set .Lmy_kernel.uses_flat_scratch, {{[01]}}
; OL:            .set .Lmy_kernel.has_dyn_sized_stack, 0
; OL:            .set .Lmy_kernel.has_recursion, 0
; OL:            .set .Lmy_kernel.has_indirect_call, 0
