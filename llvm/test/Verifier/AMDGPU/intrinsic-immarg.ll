; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1)
define void @buffer_load_f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %data0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 %bool, i1 false)
  %data0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 %bool, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %data1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 %bool)
  %data1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 %bool)
  ret void
}

declare float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32>, i32, i32, i32)
define void @raw_buffer_load_f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %arg) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg
  ; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %arg)
  %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %arg)
  ret void
}

declare float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32>, i32, i32, i32)
define void @raw_buffer_load_format_f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %arg) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg
  ; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %arg)
  %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %arg)
  ret void
}

declare float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32>, i32, i32, i32, i32)
define void @struct_buffer_load_f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %arg) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg
  ; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %arg)
  %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %arg)
  ret void
}

declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32)
define void @struct_buffer_load_format_f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %arg) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg
  ; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %arg)
  %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %arg)
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32)
define void @invalid_image_sample_1d_v4f32_f32(float %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc, i32 %dmask, i1 %bool, i32 %arg) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %dmask
  ; CHECK-NEXT: %data0 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 %dmask, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %data0 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 %dmask, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %data1 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 %bool, i32 0, i32 0)
  %data1 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 %bool, i32 0, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg
  ; CHECK-NEXT:   %data2 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 %arg, i32 0)
  %data2 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 %arg, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg
  ; CHECK-NEXT:   %data3 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 %arg)
  %data3 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 %arg)
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)
define void @exp_invalid_inputs(i32 %tgt, i32 %en, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %en
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.f32(i32 0, i32 %en, float 1.000000e+00, float 2.000000e+00, float 5.000000e-01, float 4.000000e+00, i1 true, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 %en, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %tgt
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.f32(i32 %tgt, i32 15, float 1.000000e+00, float 2.000000e+00, float 5.000000e-01, float 4.000000e+00, i1 true, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 %tgt, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.000000e+00, float 2.000000e+00, float 5.000000e-01, float 4.000000e+00, i1 %bool, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 %bool, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.000000e+00, float 2.000000e+00, float 5.000000e-01, float 4.000000e+00, i1 false, i1 %bool)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 %bool)
  ret void
}

declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1)

define void @exp_compr_invalid_inputs(i32 %tgt, i32 %en, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %en
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 %en, <2 x half> <half 0xH3C00, half 0xH4000>, <2 x half> <half 0xH3800, half 0xH4400>, i1 true, i1 false)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 %en, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %tgt
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.compr.v2f16(i32 %tgt, i32 5, <2 x half> <half 0xH3C00, half 0xH4000>, <2 x half> <half 0xH3800, half 0xH4400>, i1 true, i1 false)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 %tgt, i32 5, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 5, <2 x half> <half 0xH3C00, half 0xH4000>, <2 x half> <half 0xH3800, half 0xH4400>, i1 %bool, i1 false)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 5, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 %bool, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 5, <2 x half> <half 0xH3C00, half 0xH4000>, <2 x half> <half 0xH3800, half 0xH4400>, i1 false, i1 %bool)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 5, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 false, i1 %bool)
  ret void
}

declare i64 @llvm.amdgcn.icmp.i64.i32(i32, i32, i32)

define i64 @invalid_nonconstant_icmp_code(i32 %a, i32 %b, i32 %c) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %c
  ; CHECK-NEXT: %result = call i64 @llvm.amdgcn.icmp.i64.i32(i32 %a, i32 %b, i32 %c)
  %result = call i64 @llvm.amdgcn.icmp.i64.i32(i32 %a, i32 %b, i32 %c)
  ret i64 %result
}

declare i64 @llvm.amdgcn.fcmp.i64.f32(float, float, i32)
define i64 @invalid_nonconstant_fcmp_code(float %a, float %b, i32 %c) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %c
  ; CHECK-NEXT: %result = call i64 @llvm.amdgcn.fcmp.i64.f32(float %a, float %b, i32 %c)
  %result = call i64 @llvm.amdgcn.fcmp.i64.f32(float %a, float %b, i32 %c)
  ret i64 %result
}

declare { float, i1 } @llvm.amdgcn.div.scale.f32(float, float, i1)
define amdgpu_kernel void @test_div_scale_f32_val_undef_undef(ptr addrspace(1) %out) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i1 undef
  ; CHECK: %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float 8.000000e+00, float undef, i1 undef)
  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float 8.0, float undef, i1 undef)
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, ptr addrspace(1) %out, align 4
  ret void
}

declare void @llvm.amdgcn.init.exec(i64)
define amdgpu_ps void @init_exec(i64 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %var
  ; CHECK-NEXT: call void @llvm.amdgcn.init.exec(i64 %var)
  call void @llvm.amdgcn.init.exec(i64 %var)
  ret void
}

declare i32 @llvm.amdgcn.s.sendmsg(i32, i32)
define void @sendmsg(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.s.sendmsg(i32 %arg0, i32 %arg1)
  %val = call i32 @llvm.amdgcn.s.sendmsg(i32 %arg0, i32 %arg1)
  ret void
}

declare i32 @llvm.amdgcn.s.sendmsghalt(i32, i32)
define void @sendmsghalt(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.s.sendmsghalt(i32 %arg0, i32 %arg1)
  %val = call i32 @llvm.amdgcn.s.sendmsghalt(i32 %arg0, i32 %arg1)
  ret void
}

declare i32 @llvm.amdgcn.s.waitcnt(i32)
define void @waitcnt(i32 %arg0) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.s.waitcnt(i32 %arg0)
  %val = call i32 @llvm.amdgcn.s.waitcnt(i32 %arg0)
  ret void
}

declare i32 @llvm.amdgcn.s.getreg(i32)
define void @getreg(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.s.getreg(i32 %arg0)
  %val = call i32 @llvm.amdgcn.s.getreg(i32 %arg0)
  ret void
}

declare i32 @llvm.amdgcn.s.sleep(i32)
define void @sleep(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.s.sleep(i32 %arg0)
  %val = call i32 @llvm.amdgcn.s.sleep(i32 %arg0)
  ret void
}

declare i32 @llvm.amdgcn.s.incperflevel(i32)
define void @incperflevel(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.s.incperflevel(i32 %arg0)
  %val = call i32 @llvm.amdgcn.s.incperflevel(i32 %arg0)
  ret void
}

declare i32 @llvm.amdgcn.s.decperflevel(i32)
define void @decperflevel(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.s.decperflevel(i32 %arg0)
  %val = call i32 @llvm.amdgcn.s.decperflevel(i32 %arg0)
  ret void
}

declare i32 @llvm.amdgcn.ds.swizzle(i32, i32)
define void @ds_swizzle(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.ds.swizzle(i32 %arg0, i32 %arg1)
  %val = call i32 @llvm.amdgcn.ds.swizzle(i32 %arg0, i32 %arg1)
  ret void
}

declare i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) nocapture, i32, i32, i32, i1, i32, i1, i1)
define amdgpu_kernel void @ds_ordered_add(ptr addrspace(2) %gds, ptr addrspace(1) %out, i32 %var, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val0 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 %var, i32 0, i1 false, i32 1, i1 true, i1 true)
  %val0 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 %var, i32 0, i1 false, i32 1, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val1 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 %var, i1 false, i32 1, i1 true, i1 true)
  %val1 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 %var, i1 false, i32 1, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val2 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 %bool, i32 1, i1 true, i1 true)
  %val2 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 %bool, i32 1, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val3 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 %var, i1 true, i1 true)
  %val3 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 %var, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val4 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 %bool, i1 true)
  %val4 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 %bool, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val5 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 %bool)
  %val5 = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 %bool)
  ret void
}

declare i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) nocapture, i32, i32, i32, i1, i32, i1, i1)
define amdgpu_kernel void @ds_ordered_swap(ptr addrspace(2) %gds, ptr addrspace(1) %out, i32 %var, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val0 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 %var, i32 0, i1 false, i32 1, i1 true, i1 true)
  %val0 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 %var, i32 0, i1 false, i32 1, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val1 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 %var, i1 false, i32 1, i1 true, i1 true)
  %val1 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 %var, i1 false, i32 1, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val2 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 %bool, i32 1, i1 true, i1 true)
  %val2 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 %bool, i32 1, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val3 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 %var, i1 true, i1 true)
  %val3 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 %var, i1 true, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val4 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 %bool, i1 true)
  %val4 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 %bool, i1 true)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val5 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 %bool)
  %val5 = call i32 @llvm.amdgcn.ds.ordered.swap(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 %bool)
  ret void
}

declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i32, i32, i1)
define amdgpu_kernel void @mov_dpp_test(ptr addrspace(1) %out, i32 %in1, i32 %var, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 %var, i32 1, i32 1, i1 true)
  %val0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 %var, i32 1, i32 1, i1 1)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val1 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 1, i32 %var, i32 1, i1 true)
  %val1 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 1, i32 %var, i32 1, i1 1)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val2 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 1, i32 1, i32 %var, i1 true)
  %val2 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 1, i32 1, i32 %var, i1 1)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val3 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 1, i32 1, i32 1, i1 %bool)
  %val3 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in1, i32 1, i32 1, i32 1, i1 %bool)
  ret void
}

declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1)
define amdgpu_kernel void @update_dpp_test(ptr addrspace(1) %out, i32 %in1, i32 %in2, i32 %var, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 %var, i32 1, i32 1, i1 true)
  %val0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 %var, i32 1, i32 1, i1 1)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val1 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 %var, i32 1, i1 true)
  %val1 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 %var, i32 1, i1 1)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val2 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 %var, i1 true)
  %val2 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 %var, i1 1)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val3 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 %bool)
  %val3 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 %bool)
  ret void
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32)
define amdgpu_ps void @load_1d(<8 x i32> inreg %rsrc, i32 %s, i32 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val0 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 %var, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %val0 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 %var, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val1 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 %var, i32 0)
  %val1 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 %var, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val2 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 %var)
  %val2 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 %var)
  ret void
}

declare {<4 x float>,i32} @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32, i32, <8 x i32>, i32, i32)
define amdgpu_ps void @load_1d_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %val) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %val
  ; CHECK-NEXT: %val0 = call { <4 x float>, i32 } @llvm.amdgcn.image.load.1d.sl_v4f32i32s.i32(i32 %val, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
  %val0 = call {<4 x float>, i32} @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 %val, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %val
  ; CHECK-NEXT: %val1 = call { <4 x float>, i32 } @llvm.amdgcn.image.load.1d.sl_v4f32i32s.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 %val, i32 0)
  %val1 = call {<4 x float>, i32} @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 %val, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %val
  ; CHECK-NEXT: %val2 = call { <4 x float>, i32 } @llvm.amdgcn.image.load.1d.sl_v4f32i32s.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 1, i32 %val)
  %val2 = call {<4 x float>, i32} @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 1, i32 %val)
  ret void
}

declare {<4 x float>, i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32)
define amdgpu_ps void @sample_1d_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, ptr addrspace(1) inreg %out, float %s, i32 %var, i1 %bool) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val0 = call { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 %var, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 0)
  %val0 = call {<4 x float>, i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 %var, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %bool
  ; CHECK-NEXT: %val1 = call { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 16, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 %bool, i32 1, i32 0)
  %val1 = call {<4 x float>, i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 16, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 %bool, i32 1, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val2 = call { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 16, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 %var, i32 0)
  %val2 = call {<4 x float>, i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 16, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 %var, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val3 = call { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 %var, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 %var)
  %val3 = call {<4 x float>, i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 %var, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 %var)
  ret void
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32, i16, <8 x i32>, i32, i32)
define amdgpu_ps void @load_1d_a16(<8 x i32> inreg %rsrc, <2 x i16> %coords, i16 %s, i32 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val0 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 %var, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %val0 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 %var, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val1 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 %var, i32 0)
  %val1 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 %var, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val2 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 %var)
  %val2 = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 %var)
  ret void
}

declare i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32, <4 x i32>, i32, i32, i32)
define amdgpu_ps void @raw_buffer_atomic_swap(<4 x i32> inreg %rsrc, i32 %data, i32 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %val2 = call i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i32 %var)
  %val2 = call i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i32 %var)
  ret void
}

declare i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32, i32, <8 x i32>, i32, i32)
define amdgpu_ps void @atomic_swap_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %val) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %val
  ; CHECK-NEXT: %val0 = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 %val, i32 0)
  %val0 = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 %val, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %val
  ; CHECK-NEXT: %val1 = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 %val)
  %val1 = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 %val)
  ret void
}

declare i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #0
define amdgpu_ps void @atomic_cmpswap_1d(<8 x i32> inreg %rsrc, i32 %cmp, i32 %swap, i32 %s, i32 %val) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %val
  ; CHECK-NEXT: %val0 = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32 %cmp, i32 %swap, i32 %s, <8 x i32> %rsrc, i32 %val, i32 0)
  %val0 = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32 %cmp, i32 %swap, i32 %s, <8 x i32> %rsrc, i32 %val, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %val
  ; CHECK-NEXT: %val1 = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32 %cmp, i32 %swap, i32 %s, <8 x i32> %rsrc, i32 0, i32 %val)
  %val1 = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32 %cmp, i32 %swap, i32 %s, <8 x i32> %rsrc, i32 0, i32 %val)
  ret void
}

declare float @llvm.amdgcn.fdot2(<2 x half>, <2 x half>, float, i1)
define float @test_fdot2(<2 x half> %arg0, <2 x half> %arg1, float %arg2, i1 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg3
  ; CHECK-NEXT: %val = call float @llvm.amdgcn.fdot2(<2 x half> %arg0, <2 x half> %arg1, float %arg2, i1 %arg3)
  %val = call float @llvm.amdgcn.fdot2(<2 x half> %arg0, <2 x half> %arg1, float %arg2, i1 %arg3)
  ret float %val
}

declare i32 @llvm.amdgcn.sdot2(<2 x i16>, <2 x i16>, i32, i1)
define i32 @test_sdot2(<2 x i16> %arg0, <2 x i16> %arg1, i32 %arg2, i1 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg3
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.sdot2(<2 x i16> %arg0, <2 x i16> %arg1, i32 %arg2, i1 %arg3)
  %val = call i32 @llvm.amdgcn.sdot2(<2 x i16> %arg0, <2 x i16> %arg1, i32 %arg2, i1 %arg3)
  ret i32 %val
}

declare i32 @llvm.amdgcn.udot2(<2 x i16>, <2 x i16>, i32, i1)
define i32 @test_udot2(<2 x i16> %arg0, <2 x i16> %arg1, i32 %arg2, i1 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg3
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.udot2(<2 x i16> %arg0, <2 x i16> %arg1, i32 %arg2, i1 %arg3)
  %val = call i32 @llvm.amdgcn.udot2(<2 x i16> %arg0, <2 x i16> %arg1, i32 %arg2, i1 %arg3)
  ret i32 %val
}

declare i32 @llvm.amdgcn.sdot4(i32, i32, i32, i1)
define i32 @test_sdot4(i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg3
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.sdot4(i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3)
  %val = call i32 @llvm.amdgcn.sdot4(i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3)
  ret i32 %val
}

declare i32 @llvm.amdgcn.udot4(i32, i32, i32, i1)
define i32 @test_udot4(i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg3
  ; CHECK-NEXT: %val = call i32 @llvm.amdgcn.udot4(i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3)
  %val = call i32 @llvm.amdgcn.udot4(i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3)
  ret i32 %val
}

declare i32 @llvm.amdgcn.permlane16.i32(i32, i32, i32, i32, i1, i1)
define i32 @test_permlane16(ptr addrspace(1) %out, i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3, i1 %arg4) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg3
  ; CHECK-NEXT: %v1 = call i32 @llvm.amdgcn.permlane16.i32(i32 %arg0, i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3, i1 false)
  %v1 = call i32 @llvm.amdgcn.permlane16.i32(i32 %arg0, i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg4
  ; CHECK-NEXT: call i32 @llvm.amdgcn.permlane16.i32(i32 %v2, i32 %arg0, i32 %arg1, i32 %arg2, i1 false, i1 %arg4)
  %v2 = call i32 @llvm.amdgcn.permlane16.i32(i32 %v2, i32 %arg0, i32 %arg1, i32 %arg2, i1 false, i1 %arg4)
  ret i32 %v2
}

declare i32 @llvm.amdgcn.permlanex16.i32(i32, i32, i32, i32, i1, i1)
define i32 @test_permlanex16(ptr addrspace(1) %out, i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3, i1 %arg4) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg3
  ; CHECK-NEXT: %v1 = call i32 @llvm.amdgcn.permlanex16.i32(i32 %arg0, i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3, i1 false)
  %v1 = call i32 @llvm.amdgcn.permlanex16.i32(i32 %arg0, i32 %arg0, i32 %arg1, i32 %arg2, i1 %arg3, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg4
  ; CHECK-NEXT: call i32 @llvm.amdgcn.permlanex16.i32(i32 %v2, i32 %arg0, i32 %arg1, i32 %arg2, i1 false, i1 %arg4)
  %v2 = call i32 @llvm.amdgcn.permlanex16.i32(i32 %v2, i32 %arg0, i32 %arg1, i32 %arg2, i1 false, i1 %arg4)
  ret i32 %v2
}

declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32)
define void @test_interp_p1(float %arg0, i32 %arg1, i32 %arg2, i32 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %val0 = call float @llvm.amdgcn.interp.p1(float %arg0, i32 %arg1, i32 0, i32 0)
  %val0 = call float @llvm.amdgcn.interp.p1(float %arg0, i32 %arg1, i32 0, i32 0)
  store volatile float %val0, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %val1 = call float @llvm.amdgcn.interp.p1(float %arg0, i32 0, i32 %arg2, i32 0)
  %val1 = call float @llvm.amdgcn.interp.p1(float %arg0, i32 0, i32 %arg2, i32 0)
  store volatile float %val1, ptr addrspace(1) undef
  ret void
}

declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32)
define void @test_interp_p2(float %arg0, float %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %val0 = call float @llvm.amdgcn.interp.p2(float %arg0, float %arg1, i32 %arg2, i32 0, i32 0)

  %val0 = call float @llvm.amdgcn.interp.p2(float %arg0, float %arg1, i32 %arg2, i32 0, i32 0)
  store volatile float %val0, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg3
  ; CHECK-NEXT: %val1 = call float @llvm.amdgcn.interp.p2(float %arg0, float %arg1, i32 0, i32 %arg3, i32 0)
  %val1 = call float @llvm.amdgcn.interp.p2(float %arg0, float %arg1, i32 0, i32 %arg3, i32 0)
  store volatile float %val1, ptr addrspace(1) undef
  ret void
}

declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32)
define void @test_interp_mov(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %val0 = call float @llvm.amdgcn.interp.mov(i32 %arg0, i32 0, i32 0, i32 0)
  %val0 = call float @llvm.amdgcn.interp.mov(i32 %arg0, i32 0, i32 0, i32 0)
  store volatile float %val0, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %val1 = call float @llvm.amdgcn.interp.mov(i32 0, i32 %arg1, i32 0, i32 0)
  %val1 = call float @llvm.amdgcn.interp.mov(i32 0, i32 %arg1, i32 0, i32 0)
  store volatile float %val1, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %val2 = call float @llvm.amdgcn.interp.mov(i32 0, i32 0, i32 %arg2, i32 0)
  %val2 = call float @llvm.amdgcn.interp.mov(i32 0, i32 0, i32 %arg2, i32 0)
  store volatile float %val2, ptr addrspace(1) undef

  ret void
}

declare float @llvm.amdgcn.interp.p1.f16(float, i32, i32, i1, i32)
define void @test_interp_p1_f16(float %arg0, i32 %arg1, i32 %arg2, i1 %arg3, i32 %arg4) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT:%val0 = call float @llvm.amdgcn.interp.p1.f16(float %arg0, i32 %arg1, i32 2, i1 false, i32 %arg4)
  %val0 = call float @llvm.amdgcn.interp.p1.f16(float %arg0, i32 %arg1, i32 2, i1 0, i32 %arg4)
  store volatile float %val0, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT:i32 %arg2
  ; CHECK-NEXT:  %val1 = call float @llvm.amdgcn.interp.p1.f16(float %arg0, i32 0, i32 %arg2, i1 false, i32 %arg4)
  %val1 = call float @llvm.amdgcn.interp.p1.f16(float %arg0, i32 0, i32 %arg2, i1 0, i32 %arg4)
  store volatile float %val1, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT:i1 %arg3
  ; CHECK-NEXT:  %val2 = call float @llvm.amdgcn.interp.p1.f16(float %arg0, i32 0, i32 0, i1 %arg3, i32 %arg4)
  %val2 = call float @llvm.amdgcn.interp.p1.f16(float %arg0, i32 0, i32 0, i1 %arg3, i32 %arg4)
  store volatile float %val2, ptr addrspace(1) undef

  ret void
}

declare half @llvm.amdgcn.interp.p2.f16(float, float, i32, i32, i1, i32)
define void @test_interp_p2_f16(float %arg0, float %arg1, i32 %arg2, i32 %arg3, i1 %arg4, i32 %arg5) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %val0 = call half @llvm.amdgcn.interp.p2.f16(float %arg0, float %arg1, i32 %arg2, i32 2, i1 false, i32 %arg5)
  %val0 = call half @llvm.amdgcn.interp.p2.f16(float %arg0, float %arg1, i32 %arg2, i32 2, i1 false, i32 %arg5)
  store volatile half %val0, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg3
  ; CHECK-NEXT: %val1 = call half @llvm.amdgcn.interp.p2.f16(float %arg0, float %arg1, i32 0, i32 %arg3, i1 false, i32 %arg5)
  %val1 = call half @llvm.amdgcn.interp.p2.f16(float %arg0, float %arg1, i32 0, i32 %arg3, i1 false, i32 %arg5)
  store volatile half %val1, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %arg4
  ; CHECK-NEXT: %val2 = call half @llvm.amdgcn.interp.p2.f16(float %arg0, float %arg1, i32 0, i32 0, i1 %arg4, i32 %arg5)
  %val2 = call half @llvm.amdgcn.interp.p2.f16(float %arg0, float %arg1, i32 0, i32 0, i1 %arg4, i32 %arg5)
  store volatile half %val2, ptr addrspace(1) undef

  ret void
}

declare <32 x i32> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x i32>, i32, i32, i32)
define void @test_mfma_f32_32x32x1f32(float %arg0, float %arg1, <32 x i32> %arg2, i32 %arg3, i32 %arg4, i32 %arg5) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg3
  ; CHECK-NEXT: %val0 = call <32 x i32> @llvm.amdgcn.mfma.f32.32x32x1f32(float %arg0, float %arg1, <32 x i32> %arg2, i32 %arg3, i32 2, i32 3)
  %val0 = call <32 x i32> @llvm.amdgcn.mfma.f32.32x32x1f32(float %arg0, float %arg1, <32 x i32> %arg2, i32 %arg3, i32 2, i32 3)
  store volatile <32 x i32> %val0, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg4
  ; CHECK-NEXT: %val1 = call <32 x i32> @llvm.amdgcn.mfma.f32.32x32x1f32(float %arg0, float %arg1, <32 x i32> %arg2, i32 1, i32 %arg4, i32 3)
  %val1 = call <32 x i32> @llvm.amdgcn.mfma.f32.32x32x1f32(float %arg0, float %arg1, <32 x i32> %arg2, i32 1, i32 %arg4, i32 3)
  store volatile <32 x i32> %val1, ptr addrspace(1) undef

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg5
  ; CHECK-NEXT: %val2 = call <32 x i32> @llvm.amdgcn.mfma.f32.32x32x1f32(float %arg0, float %arg1, <32 x i32> %arg2, i32 1, i32 2, i32 %arg5)
  %val2 = call <32 x i32> @llvm.amdgcn.mfma.f32.32x32x1f32(float %arg0, float %arg1, <32 x i32> %arg2, i32 1, i32 2, i32 %arg5)
  store volatile <32 x i32> %val2, ptr addrspace(1) undef

  ret void
}

declare void @llvm.amdgcn.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i1)
define amdgpu_cs void @test_buffer_atomic_fadd(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 %offset, i1 %slc) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %slc
  ; CHECK-ENXT: call void @llvm.amdgcn.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %offset, i1 %slc)
  call void @llvm.amdgcn.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %offset, i1 %slc)
  ret void
}
