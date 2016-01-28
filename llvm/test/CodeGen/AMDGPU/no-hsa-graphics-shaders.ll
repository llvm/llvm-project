; RUN: not llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa < %s 2>&1 | FileCheck %s

; CHECK: error: unsupported non-compute shaders with HSA in pixel_shader
define void @pixel_shader() #0 {
  ret void
}

define void @vertex_shader() #1 {
  ret void
}

define void @geometry_shader() #2 {
  ret void
}

attributes #0 = { nounwind "ShaderType"="0" }
attributes #1 = { nounwind "ShaderType"="1" }
attributes #2 = { nounwind "ShaderType"="2" }
