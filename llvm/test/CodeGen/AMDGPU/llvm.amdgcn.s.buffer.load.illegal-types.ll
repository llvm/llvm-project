; RUN: not llc -mtriple=amdgcn -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SDAG
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GISEL
; RUN: not llc -mtriple=amdgcn -mcpu=gfx1100 -filetype=null %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SDAG
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1100 -filetype=null %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GISEL

; There is no s_buffer_load for these result types.

; CHECK: error: {{.*}} in function s_buffer_load_i1 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_i1(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call i1 @llvm.amdgcn.s.buffer.load.i1(<4 x i32> %desc, i32 0, i32 0)
  store i1 %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_i4 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_i4(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call i4 @llvm.amdgcn.s.buffer.load.i4(<4 x i32> %desc, i32 0, i32 0)
  store i4 %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_v2i1 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_v2i1(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call <2 x i1> @llvm.amdgcn.s.buffer.load.v2i1(<4 x i32> %desc, i32 0, i32 0)
  store <2 x i1> %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_v3i16 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_v3i16(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call <3 x i16> @llvm.amdgcn.s.buffer.load.v3i16(<4 x i32> %desc, i32 0, i32 0)
  store <3 x i16> %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_v3f16 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_v3f16(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call <3 x half> @llvm.amdgcn.s.buffer.load.v3f16(<4 x i32> %desc, i32 0, i32 0)
  store <3 x half> %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_v6i8 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_v6i8(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call <6 x i8> @llvm.amdgcn.s.buffer.load.v6i8(<4 x i32> %desc, i32 0, i32 0)
  store <6 x i8> %load, ptr addrspace(1) %out
  ret void
}

; i128 is illegal for SelectionDAG, but GlobalISel selects s_buffer_load_b128.
; GISEL-NOT: in function {{.*}}_i128
; SDAG: error: {{.*}} in function s_buffer_load_i128 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_i128(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call i128 @llvm.amdgcn.s.buffer.load.i128(<4 x i32> %desc, i32 0, i32 0)
  store i128 %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_i8 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_i8(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call i8 @llvm.amdgcn.s.buffer.load.i8(<4 x i32> %desc, i32 0, i32 0)
  store i8 %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_i16 {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_i16(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %load = call i16 @llvm.amdgcn.s.buffer.load.i16(<4 x i32> %desc, i32 0, i32 0)
  store i16 %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_i1_divergent {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_i1_divergent(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %offset = call i32 @llvm.amdgcn.workitem.id.x()
  %load = call i1 @llvm.amdgcn.s.buffer.load.i1(<4 x i32> %desc, i32 %offset, i32 0)
  store i1 %load, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}} in function s_buffer_load_v6i8_divergent {{.*}}: unsupported s_buffer_load result type
define amdgpu_ps void @s_buffer_load_v6i8_divergent(<4 x i32> inreg %desc, ptr addrspace(1) %out) {
  %offset = call i32 @llvm.amdgcn.workitem.id.x()
  %load = call <6 x i8> @llvm.amdgcn.s.buffer.load.v6i8(<4 x i32> %desc, i32 %offset, i32 0)
  store <6 x i8> %load, ptr addrspace(1) %out
  ret void
}
