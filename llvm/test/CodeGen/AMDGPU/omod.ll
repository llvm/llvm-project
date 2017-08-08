; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; IEEE bit enabled for compute kernel, no shouldn't use.
; GCN-LABEL: {{^}}v_omod_div2_f32_enable_ieee_signed_zeros:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, [[A]]{{$}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0.5, [[ADD]]{{$}}
define amdgpu_kernel void @v_omod_div2_f32_enable_ieee_signed_zeros(float addrspace(1)* %out, float addrspace(1)* %aptr) #4 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 0.5
  store float %div2, float addrspace(1)* %out.gep
  ret void
}

; IEEE bit enabled for compute kernel, no shouldn't use even though nsz is allowed
; GCN-LABEL: {{^}}v_omod_div2_f32_enable_ieee_nsz:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, [[A]]{{$}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0.5, [[ADD]]{{$}}
define amdgpu_kernel void @v_omod_div2_f32_enable_ieee_nsz(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 0.5
  store float %div2, float addrspace(1)* %out.gep
  ret void
}

; Only allow without IEEE bit if signed zeros are significant.
; GCN-LABEL: {{^}}v_omod_div2_f32_signed_zeros:
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, v0{{$}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0.5, [[ADD]]{{$}}
define amdgpu_ps void @v_omod_div2_f32_signed_zeros(float %a) #4 {
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 0.5
  store float %div2, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_div2_f32:
; GCN: v_add_f32_e64 v{{[0-9]+}}, v0, 1.0 div:2{{$}}
define amdgpu_ps void @v_omod_div2_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 0.5
  store float %div2, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_mul2_f32:
; GCN: v_add_f32_e64 v{{[0-9]+}}, v0, 1.0 mul:2{{$}}
define amdgpu_ps void @v_omod_mul2_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 2.0
  store float %div2, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_mul4_f32:
; GCN: v_add_f32_e64 v{{[0-9]+}}, v0, 1.0 mul:4{{$}}
define amdgpu_ps void @v_omod_mul4_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 4.0
  store float %div2, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_mul4_multi_use_f32:
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, v0{{$}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 4.0, [[ADD]]{{$}}
define amdgpu_ps void @v_omod_mul4_multi_use_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 4.0
  store float %div2, float addrspace(1)* undef
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_mul4_dbg_use_f32:
; GCN: v_add_f32_e64 v{{[0-9]+}}, v0, 1.0 mul:4{{$}}
define amdgpu_ps void @v_omod_mul4_dbg_use_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  call void @llvm.dbg.value(metadata float %add, i64 0, metadata !4, metadata !9), !dbg !10
  %div2 = fmul float %add, 4.0
  store float %div2, float addrspace(1)* undef
  ret void
}

; Clamp is applied after omod, folding both into instruction is OK.
; GCN-LABEL: {{^}}v_clamp_omod_div2_f32:
; GCN: v_add_f32_e64 v{{[0-9]+}}, v0, 1.0 clamp div:2{{$}}
define amdgpu_ps void @v_clamp_omod_div2_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 0.5

  %max = call float @llvm.maxnum.f32(float %div2, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* undef
  ret void
}

; Cannot fold omod into clamp
; GCN-LABEL: {{^}}v_omod_div2_clamp_f32:
; GCN: v_add_f32_e64 [[ADD:v[0-9]+]], v0, 1.0 clamp{{$}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0.5, [[ADD]]{{$}}
define amdgpu_ps void @v_omod_div2_clamp_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  %div2 = fmul float %clamp, 0.5
  store float %div2, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_div2_abs_src_f32:
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, v0{{$}}
; GCN: v_mul_f32_e64 v{{[0-9]+}}, |[[ADD]]|, 0.5{{$}}
define amdgpu_ps void @v_omod_div2_abs_src_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %abs.add = call float @llvm.fabs.f32(float %add)
  %div2 = fmul float %abs.add, 0.5
  store float %div2, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_add_self_clamp_f32:
; GCN: v_add_f32_e64 v{{[0-9]+}}, v0, v0 clamp{{$}}
define amdgpu_ps void @v_omod_add_self_clamp_f32(float %a) #0 {
  %add = fadd float %a, %a
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_add_clamp_self_f32:
; GCN: v_max_f32_e64 [[CLAMP:v[0-9]+]], v0, v0 clamp{{$}}
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[CLAMP]], [[CLAMP]]{{$}}
define amdgpu_ps void @v_omod_add_clamp_self_f32(float %a) #0 {
  %max = call float @llvm.maxnum.f32(float %a, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  %add = fadd float %clamp, %clamp
  store float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_add_abs_self_f32:
; GCN: v_add_f32_e32 [[X:v[0-9]+]], 1.0, v0
; GCN: v_add_f32_e64 v{{[0-9]+}}, |[[X]]|, |[[X]]|{{$}}
define amdgpu_ps void @v_omod_add_abs_self_f32(float %a) #0 {
  %x = fadd float %a, 1.0
  %abs.x = call float @llvm.fabs.f32(float %x)
  %add = fadd float %abs.x, %abs.x
  store float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_add_abs_x_x_f32:

; GCN: v_add_f32_e32 [[X:v[0-9]+]], 1.0, v0
; GCN: v_add_f32_e64 v{{[0-9]+}}, |[[X]]|, [[X]]{{$}}
define amdgpu_ps void @v_omod_add_abs_x_x_f32(float %a) #0 {
  %x = fadd float %a, 1.0
  %abs.x = call float @llvm.fabs.f32(float %x)
  %add = fadd float %abs.x, %x
  store float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_add_x_abs_x_f32:
; GCN: v_add_f32_e32 [[X:v[0-9]+]], 1.0, v0
; GCN: v_add_f32_e64 v{{[0-9]+}}, [[X]], |[[X]]|{{$}}
define amdgpu_ps void @v_omod_add_x_abs_x_f32(float %a) #0 {
  %x = fadd float %a, 1.0
  %abs.x = call float @llvm.fabs.f32(float %x)
  %add = fadd float %x, %abs.x
  store float %add, float addrspace(1)* undef
  ret void
}

; Don't fold omod into omod into another omod.
; GCN-LABEL: {{^}}v_omod_div2_omod_div2_f32:
; GCN: v_add_f32_e64 [[ADD:v[0-9]+]], v0, 1.0 div:2{{$}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0.5, [[ADD]]{{$}}
define amdgpu_ps void @v_omod_div2_omod_div2_f32(float %a) #0 {
  %add = fadd float %a, 1.0
  %div2.0 = fmul float %add, 0.5
  %div2.1 = fmul float %div2.0, 0.5
  store float %div2.1, float addrspace(1)* undef
  ret void
}

; Don't fold omod if denorms enabled
; GCN-LABEL: {{^}}v_omod_div2_f32_denormals:
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, v0{{$}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0.5, [[ADD]]{{$}}
define amdgpu_ps void @v_omod_div2_f32_denormals(float %a) #2 {
  %add = fadd float %a, 1.0
  %div2 = fmul float %add, 0.5
  store float %div2, float addrspace(1)* undef
  ret void
}

; Don't fold omod if denorms enabled for add form.
; GCN-LABEL: {{^}}v_omod_mul2_f32_denormals:
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, v0{{$}}
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[ADD]], [[ADD]]{{$}}
define amdgpu_ps void @v_omod_mul2_f32_denormals(float %a) #2 {
  %add = fadd float %a, 1.0
  %mul2 = fadd float %add, %add
  store float %mul2, float addrspace(1)* undef
  ret void
}

; Don't fold omod if denorms enabled
; GCN-LABEL: {{^}}v_omod_div2_f16_denormals:
; VI: v_add_f16_e32 [[ADD:v[0-9]+]], 1.0, v0{{$}}
; VI: v_mul_f16_e32 v{{[0-9]+}}, 0.5, [[ADD]]{{$}}
define amdgpu_ps void @v_omod_div2_f16_denormals(half %a) #0 {
  %add = fadd half %a, 1.0
  %div2 = fmul half %add, 0.5
  store half %div2, half addrspace(1)* undef
  ret void
}

; Don't fold omod if denorms enabled for add form.
; GCN-LABEL: {{^}}v_omod_mul2_f16_denormals:
; VI: v_add_f16_e32 [[ADD:v[0-9]+]], 1.0, v0{{$}}
; VI: v_add_f16_e32 v{{[0-9]+}}, [[ADD]], [[ADD]]{{$}}
define amdgpu_ps void @v_omod_mul2_f16_denormals(half %a) #0 {
  %add = fadd half %a, 1.0
  %mul2 = fadd half %add, %add
  store half %mul2, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_div2_f16_no_denormals:
; VI-NOT: v0
; VI: v_add_f16_e64 [[ADD:v[0-9]+]], v0, 1.0 div:2{{$}}
define amdgpu_ps void @v_omod_div2_f16_no_denormals(half %a) #3 {
  %add = fadd half %a, 1.0
  %div2 = fmul half %add, 0.5
  store half %div2, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_omod_mac_to_mad:
; GCN: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]}} mul:2{{$}}
define amdgpu_ps void @v_omod_mac_to_mad(float %b, float %a) #0 {
  %mul = fmul float %a, %a
  %add = fadd float %mul, %b
  %mad = fmul float %add, 2.0
  %res = fmul float %mad, %b
  store float %res, float addrspace(1)* undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fabs.f32(float) #1
declare float @llvm.floor.f32(float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1
declare float @llvm.amdgcn.fmed3.f32(float, float, float) #1
declare double @llvm.fabs.f64(double) #1
declare double @llvm.minnum.f64(double, double) #1
declare double @llvm.maxnum.f64(double, double) #1
declare half @llvm.fabs.f16(half) #1
declare half @llvm.minnum.f16(half, half) #1
declare half @llvm.maxnum.f16(half, half) #1
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "no-signed-zeros-fp-math"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "target-features"="+fp32-denormals" "no-signed-zeros-fp-math"="true" }
attributes #3 = { nounwind "target-features"="-fp64-fp16-denormals" "no-signed-zeros-fp-math"="true" }
attributes #4 = { nounwind "no-signed-zeros-fp-math"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "/tmp/foo.cl", directory: "/dev/null")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "add", arg: 1, scope: !5, file: !1, line: 1)
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "float", size: 32, align: 32)
!9 = !DIExpression()
!10 = !DILocation(line: 1, column: 42, scope: !5)
