; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx900 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx900 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.write_register
define amdgpu_kernel void @test_invalid_write_register_i32() nounwind {
  call void @llvm.write_register.i32(metadata !0, i32 0)
  ret void
}

!0 = !{!"not-a-register"}
