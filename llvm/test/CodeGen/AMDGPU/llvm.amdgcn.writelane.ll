; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,CIGFX9 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx802 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,CIGFX9 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,GFX10 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx1100 -verify-machineinstrs -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=CHECK,GFX10 %s

declare i32 @llvm.amdgcn.writelane(i32, i32, i32) #0
declare i64 @llvm.amdgcn.writelane.i64(i64, i32, i64) #0
declare double @llvm.amdgcn.writelane.f64(double, i32, double) #0

; CHECK-LABEL: {{^}}test_writelane_sreg_i32:
; CIGFX9: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg_i32(ptr addrspace(1) %out, i32 %src0, i32 %src1) #1 {
  %oldval = load i32, ptr addrspace(1) %out
  %writelane = call i32 @llvm.amdgcn.writelane.i32(i32 %src0, i32 %src1, i32 %oldval)
  store i32 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_sreg_i64:
; CIGFX9: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, m0
; CIGFX9-NEXT: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GFX10-NEXT: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg_i64(ptr addrspace(1) %out, i64 %src0, i32 %src1) #1 {
  %oldval = load i64, ptr addrspace(1) %out
  %writelane = call i64 @llvm.amdgcn.writelane.i64(i64 %src0, i32 %src1, i64 %oldval)
  store i64 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_sreg_f64:
; CIGFX9: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, m0
; CIGFX9-NEXT: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GFX10-NEXT: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg_f64(ptr addrspace(1) %out, double %src0, i32 %src1) #1 {
  %oldval = load double, ptr addrspace(1) %out
  %writelane = call double @llvm.amdgcn.writelane.f64(double %src0, i32 %src1, double %oldval)
  store double %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_sreg_i32:
; CHECK: v_writelane_b32 v{{[0-9]+}}, 32, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_sreg_i32(ptr addrspace(1) %out, i32 %src1) #1 {
  %oldval = load i32, ptr addrspace(1) %out
  %writelane = call i32 @llvm.amdgcn.writelane.i32(i32 32, i32 %src1, i32 %oldval)
  store i32 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_sreg_i64:
; CHECK: v_writelane_b32 v{{[0-9]+}}, 32, s{{[0-9]+}}
; CHECK-NEXT: v_writelane_b32 v{{[0-9]+}}, 0, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_sreg_i64(ptr addrspace(1) %out, i32 %src1) #1 {
  %oldval = load i64, ptr addrspace(1) %out
  %writelane = call i64 @llvm.amdgcn.writelane.i64(i64 32, i32 %src1, i64 %oldval)
  store i64 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: fold both SGPR's
; CHECK-LABEL: {{^}}test_writelane_imm_sreg_f64:
; CHECK: s_mov_b32 [[SGPR:s[0-9]+]], 0x40400000
; CIGFX9: v_writelane_b32 v{{[0-9]+}}, 0, m0
; CIGFX9-NEXT: v_writelane_b32 v{{[0-9]+}}, [[SGPR]], m0
; GFX10: v_writelane_b32 v{{[0-9]+}}, 0, s{{[0-9]+}}
; GFX10-NEXT: v_writelane_b32 v{{[0-9]+}}, [[SGPR]], s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_sreg_f64(ptr addrspace(1) %out, i32 %src1) #1 {
  %oldval = load double, ptr addrspace(1) %out
  %writelane = call double @llvm.amdgcn.writelane.f64(double 32.0, i32 %src1, double %oldval)
  store double %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_vreg_lane_i32:
; CHECK: v_readfirstlane_b32 [[LANE:s[0-9]+]], v{{[0-9]+}}
; CHECK: v_writelane_b32 v{{[0-9]+}}, 12, [[LANE]]
define amdgpu_kernel void @test_writelane_vreg_lane_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x i32>, ptr addrspace(1) %in, i32 %tid
  %args = load <2 x i32>, ptr addrspace(1) %gep.in
  %oldval = load i32, ptr addrspace(1) %out
  %lane = extractelement <2 x i32> %args, i32 1
  %writelane = call i32 @llvm.amdgcn.writelane.i32(i32 12, i32 %lane, i32 %oldval)
  store i32 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_vreg_lane_i64:
; CHECK: v_readfirstlane_b32 [[LANE:s[0-9]+]], v{{[0-9]+}}
; CHECK: v_writelane_b32 v{{[0-9]+}}, 12, [[LANE]]
; CHECK-NEXT: v_writelane_b32 v{{[0-9]+}}, 0, [[LANE]]
define amdgpu_kernel void @test_writelane_vreg_lane_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x i64>, ptr addrspace(1) %in, i32 %tid
  %args = load <2 x i64>, ptr addrspace(1) %gep.in
  %oldval = load i64, ptr addrspace(1) %out
  %lane = extractelement <2 x i64> %args, i32 1
  %lane32 = trunc i64 %lane to i32
  %writelane = call i64 @llvm.amdgcn.writelane.i64(i64 12, i32 %lane32, i64 %oldval)
  store i64 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: fold both SGPR's
; CHECK-LABEL: {{^}}test_writelane_vreg_lane_f64:
; CHECK: s_mov_b32 [[SGPR:s[0-9]+]], 0x40280000
; CHECK: v_readfirstlane_b32 [[LANE:.*]], v{{[0-9]+}}
; CHECK: v_writelane_b32 v{{[0-9]+}}, 0, [[LANE]]
; CHECK-NEXT: v_writelane_b32 v{{[0-9]+}}, [[SGPR]], [[LANE]]
define amdgpu_kernel void @test_writelane_vreg_lane_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x double>, ptr addrspace(1) %in, i32 %tid
  %args = load <2 x double>, ptr addrspace(1) %gep.in
  %oldval = load double, ptr addrspace(1) %out
  %lane = extractelement <2 x double> %args, i32 1
  %lane_cast = bitcast double %lane to i64
  %lane32 = trunc i64 %lane_cast to i32
  %writelane = call double @llvm.amdgcn.writelane.f64(double 12.0, i32 %lane32, double %oldval)
  store double %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_m0_sreg_i32:
; CHECK: s_mov_b32 m0, -1
; CIGFX9: s_mov_b32 [[COPY_M0:s[0-9]+]], m0
; CIGFX9: v_writelane_b32 v{{[0-9]+}}, [[COPY_M0]], m0
; GFX10: v_writelane_b32 v{{[0-9]+}}, m0, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_m0_sreg_i32(ptr addrspace(1) %out, i32 %src1) #1 {
  %oldval = load i32, ptr addrspace(1) %out
  %m0 = call i32 asm "s_mov_b32 m0, -1", "={m0}"()
  %writelane = call i32 @llvm.amdgcn.writelane.i32(i32 %m0, i32 %src1, i32 %oldval)
  store i32 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_i32:
; CHECK: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, 32
define amdgpu_kernel void @test_writelane_imm_i32(ptr addrspace(1) %out, i32 %src0) #1 {
  %oldval = load i32, ptr addrspace(1) %out
  %writelane = call i32 @llvm.amdgcn.writelane.i32(i32 %src0, i32 32, i32 %oldval) #0
  store i32 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_i64:
; CHECK: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, 32
; CHECK-NEXT: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, 32
define amdgpu_kernel void @test_writelane_imm_i64(ptr addrspace(1) %out, i64 %src0) #1 {
  %oldval = load i64, ptr addrspace(1) %out
  %writelane = call i64 @llvm.amdgcn.writelane.i64(i64 %src0, i32 32, i64 %oldval) #0
  store i64 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_f64:
; CHECK: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, 32
; CHECK-NEXT: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, 32
define amdgpu_kernel void @test_writelane_imm_f64(ptr addrspace(1) %out, double %src0) #1 {
  %oldval = load double, ptr addrspace(1) %out
  %writelane = call double @llvm.amdgcn.writelane.f64(double %src0, i32 32, double %oldval) #0
  store double %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_sreg_oldval_i32:
; CHECK: v_mov_b32_e32 [[OLDVAL:v[0-9]+]], s{{[0-9]+}}
; CIGFX9: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg_oldval_i32(i32 inreg %oldval, ptr addrspace(1) %out, i32 %src0, i32 %src1) #1 {
  %writelane = call i32 @llvm.amdgcn.writelane.i32(i32 %src0, i32 %src1, i32 %oldval)
  store i32 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_sreg_oldval_i64:
; CHECK: v_mov_b32_e32 [[OLDSUB0:v[0-9]+]], s{{[0-9]+}}
; CHECK: v_mov_b32_e32 [[OLDSUB1:v[0-9]+]], s{{[0-9]+}}
; CIGFX9: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, m0
; CIGFX9-NEXT: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg_oldval_i64(i64 inreg %oldval, ptr addrspace(1) %out, i64 %src0, i32 %src1) #1 {
  %writelane = call i64 @llvm.amdgcn.writelane.i64(i64 %src0, i32 %src1, i64 %oldval)
  store i64 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_sreg_oldval_f64:
; CHECK: v_mov_b32_e32 [[OLDSUB0:v[0-9]+]], s{{[0-9]+}}
; CHECK: v_mov_b32_e32 [[OLDSUB1:v[0-9]+]], s{{[0-9]+}}
; CIGFX9: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, m0
; CIGFX9-NEXT: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg_oldval_f64(double inreg %oldval, ptr addrspace(1) %out, double %src0, i32 %src1) #1 {
  %writelane = call double @llvm.amdgcn.writelane.f64(double %src0, i32 %src1, double %oldval)
  store double %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_oldval_i32:
; CHECK: v_mov_b32_e32 [[OLDVAL:v[0-9]+]], 42
; CIGFX9: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_oldval_i32(ptr addrspace(1) %out, i32 %src0, i32 %src1) #1 {
  %writelane = call i32 @llvm.amdgcn.writelane.i32(i32 %src0, i32 %src1, i32 42)
  store i32 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_oldval_i64:
; CHECK: v_mov_b32_e32 [[OLDSUB0:v[0-9]+]], 42
; CHECK: v_mov_b32_e32 [[OLDSUB1:v[0-9]+]], 0
; CIGFX9: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, m0
; CIGFX9-NEXT: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10-NEXT: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_oldval_i64(ptr addrspace(1) %out, i64 %src0, i32 %src1) #1 {
  %writelane = call i64 @llvm.amdgcn.writelane.i64(i64 %src0, i32 %src1, i64 42)
  store i64 %writelane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_oldval_f64:
; CHECK: v_mov_b32_e32 [[OLDSUB0:v[0-9]+]], 0
; CHECK: v_mov_b32_e32 [[OLDSUB1:v[0-9]+]], 0x40450000
; CIGFX9: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, m0
; CIGFX9-NEXT: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDSUB0]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10-NEXT: v_writelane_b32 [[OLDSUB1]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_oldval_f64(ptr addrspace(1) %out, double %src0, i32 %src1) #1 {
  %writelane = call double @llvm.amdgcn.writelane.f64(double %src0, i32 %src1, double 42.0)
  store double %writelane, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }
