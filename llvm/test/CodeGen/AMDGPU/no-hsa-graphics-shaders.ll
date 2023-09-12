; RUN: not llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa < %s 2>&1 | FileCheck %s

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

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
