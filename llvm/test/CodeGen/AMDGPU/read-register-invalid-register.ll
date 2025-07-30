; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx900 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx900 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s

declare i32 @llvm.read_register.i32(metadata) #0

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_i1(ptr addrspace(1) %out) nounwind {
  %reg = call i1 @llvm.read_register.i1(metadata !0)
  store i1 %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_i16(ptr addrspace(1) %out) nounwind {
  %reg = call i16 @llvm.read_register.i16(metadata !0)
  store i16 %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_i32(ptr addrspace(1) %out) nounwind {
  %reg = call i32 @llvm.read_register.i32(metadata !0)
  store i32 %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_i64(ptr addrspace(1) %out) nounwind {
  %reg = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v2i32(ptr addrspace(1) %out) nounwind {
  %reg = call <2 x i32> @llvm.read_register.v2i32(metadata !0)
  store <2 x i32> %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v3i32(ptr addrspace(1) %out) nounwind {
  %reg = call <3 x i32> @llvm.read_register.v3i32(metadata !0)
  store <3 x i32> %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v4i32(ptr addrspace(1) %out) nounwind {
  %reg = call <4 x i32> @llvm.read_register.v4i32(metadata !0)
  store <4 x i32> %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v5i32(ptr addrspace(1) %out) nounwind {
  %reg = call <5 x i32> @llvm.read_register.v5i32(metadata !0)
  store <5 x i32> %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v6i32(ptr addrspace(1) %out) nounwind {
  %reg = call <6 x i32> @llvm.read_register.v6i32(metadata !0)
  store <6 x i32> %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v8i32(ptr addrspace(1) %out) nounwind {
  %reg = call <8 x i32> @llvm.read_register.v8i32(metadata !0)
  store <8 x i32> %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v16i32(ptr addrspace(1) %out) nounwind {
  %reg = call <16 x i32> @llvm.read_register.v16i32(metadata !0)
  store <16 x i32> %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "not-a-register" for llvm.read_register
define amdgpu_kernel void @test_invalid_register_v32i32(ptr addrspace(1) %out) nounwind {
  %reg = call <32 x i32> @llvm.read_register.v32i32(metadata !0)
  store <32 x i32> %reg, ptr addrspace(1) %out
  ret void
}

!0 = !{!"not-a-register"}
