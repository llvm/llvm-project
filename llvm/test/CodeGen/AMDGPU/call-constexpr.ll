; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-fix-function-bitcasts < %s | FileCheck -check-prefix=OPT %s

; GCN-LABEL: {{^}}test_bitcast_return_type_noinline:
; GCN: s_getpc_b64
; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ret_i32_noinline@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, ret_i32_noinline@rel32@hi+12
; GCN: s_swappc_b64
; OPT-LABEL: @test_bitcast_return_type_noinline(
; OPT: %val = call i32 @ret_i32_noinline()
; OPT: bitcast i32 %val to float
define amdgpu_kernel void @test_bitcast_return_type_noinline() #0 {
  %val = call float bitcast (i32()* @ret_i32_noinline to float()*)()
  %op = fadd float %val, 1.0
  store volatile float %op, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_bitcast_return_type_alwaysinline:
; GCN-NOT: s_getpc_b64
; GCN-NOT: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ret_i32_alwaysinline@rel32@lo+4
; GCN-NOT: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, ret_i32_alwaysinline@rel32@hi+12
; GCN-NOT: s_swappc_b64
; OPT-LABEL: @test_bitcast_return_type_alwaysinline(
; OPT: %val = call i32 @ret_i32_alwaysinline()
; OPT: bitcast i32 %val to float
define amdgpu_kernel void @test_bitcast_return_type_alwaysinline() #0 {
  %val = call float bitcast (i32()* @ret_i32_alwaysinline to float()*)()
  %op = fadd float %val, 1.0
  store volatile float %op, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_bitcast_argument_type:
; GCN: s_getpc_b64
; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ident_i32@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, ident_i32@rel32@hi+12
; GCN: s_swappc_b64
; OPT-LABEL: @test_bitcast_argument_type(
; OPT: %1 = bitcast float 2.000000e+00 to i32
; OPT: %val = call i32 @ident_i32(i32 %1)
; OPT-NOT: bitcast i32 %val to float
define amdgpu_kernel void @test_bitcast_argument_type() #0 {
  %val = call i32 bitcast (i32(i32)* @ident_i32 to i32(float)*)(float 2.0)
  %op = add i32 %val, 1
  store volatile i32 %op, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_bitcast_argument_and_return_types:
; GCN: s_getpc_b64
; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ident_i32@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, ident_i32@rel32@hi+12
; GCN: s_swappc_b64
; OPT-LABEL: @test_bitcast_argument_and_return_types(
; OPT: %1 = bitcast float 2.000000e+00 to i32
; OPT: %val = call i32 @ident_i32(i32 %1)
; OPT: bitcast i32 %val to float
define amdgpu_kernel void @test_bitcast_argument_and_return_types() #0 {
  %val = call float bitcast (i32(i32)* @ident_i32 to float(float)*)(float 2.0)
  %op = fadd float %val, 1.0
  store volatile float %op, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}use_workitem_id_x:
; GCN: s_waitcnt
; GCN-NEXT: v_and_b32_e32 [[TMP:v[0-9]+]], 0x3ff, v31
; GCN-NEXT: v_add_i32_e32 v0, vcc, [[TMP]], v0
; GCN-NEXT: s_setpc_b64
define hidden i32 @use_workitem_id_x(i32 %arg0) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %op = add i32 %id, %arg0
  ret i32 %op
}

; GCN-LABEL: {{^}}test_bitcast_use_workitem_id_x:
; GCN: v_mov_b32_e32 v31, v0
; GCN: s_getpc_b64
; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, use_workitem_id_x@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, use_workitem_id_x@rel32@hi+12
; GCN: v_mov_b32_e32 v0, 9
; GCN: s_swappc_b64
; GCN: v_add_f32_e32
; OPT-LABEL: @use_workitem_id_x(
; OPT: %val = call i32 @use_workitem_id_x(i32 9)
; OPT: bitcast i32 %val to float
define amdgpu_kernel void @test_bitcast_use_workitem_id_x() #0 {
  %val = call float bitcast (i32(i32)* @use_workitem_id_x to float(i32)*)(i32 9)
  %op = fadd float %val, 1.0
  store volatile float %op, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_invoke:
; GCN: s_getpc_b64
; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ident_i32@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, ident_i32@rel32@hi+12
; GCN: s_swappc_b64
; OPT-LABEL: @test_invoke(
; OPT: %1 = bitcast float 2.000000e+00 to i32
; OPT: %val = invoke i32 @ident_i32(i32 %1)
; OPT-NEXT: to label %continue.split unwind label %broken
; OPT-LABEL: continue.split:
; OPT: bitcast i32 %val to float
@_ZTIi = external global i8*
declare i32 @__gxx_personality_v0(...)
define amdgpu_kernel void @test_invoke() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %val = invoke float bitcast (i32(i32)* @ident_i32 to float(float)*)(float 2.0)
          to label %continue unwind label %broken

broken:
  landingpad { i8*, i32 } catch i8** @_ZTIi
  ret void

continue:
  %op = fadd float %val, 1.0
  store volatile float %op, float addrspace(1)* undef
  ret void
}

; Callees appears last in source file to test that we still lower their
; arguments before we lower any calls to them.

define hidden i32 @ret_i32_noinline() #0 {
  ret i32 4
}

define hidden i32 @ret_i32_alwaysinline() #1 {
  ret i32 4
}

define hidden i32 @ident_i32(i32 %i) #0 {
  ret i32 %i
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind noinline }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { nounwind readnone speculatable }
