; RUN: llc -O0 -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - %s

define amdgpu_kernel void @test_long_add4() {
entry:
  %add = add <4 x i64> zeroinitializer, zeroinitializer
  store <4 x i64> %add, <4 x i64> addrspace(1)* null, align 32
  ret void
}
