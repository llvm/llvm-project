; RUN: not llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=2 < %s 2>&1 | FileCheck %s

; CHECK: in function pixel_s{{.*}}: unsupported non-compute shaders with HSA
define amdgpu_ps void @pixel_shader() #0 {
  ret void
}

; CHECK: in function vertex_s{{.*}}: unsupported non-compute shaders with HSA
define amdgpu_vs void @vertex_shader() #0 {
  ret void
}

; CHECK: in function geometry_s{{.*}}: unsupported non-compute shaders with HSA
define amdgpu_gs void @geometry_shader() #0 {
  ret void
}
