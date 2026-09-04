; RUN: not opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=amdgpu-preload-kernel-arguments -disable-output < %s 2>&1 | FileCheck %s

; CHECK: error: invalid amdgpu kernarg preload argument range
define amdgpu_kernel void @invalid_range(i32 %arg) #0 {
  ret void
}

; CHECK: error: incomplete amdgpu kernarg preload range
define amdgpu_kernel void @incomplete_range(i32 %arg) #1 {
  ret void
}

; CHECK: error: amdgpu kernarg preload range exceeds available user SGPRs
define amdgpu_kernel void @range_too_large(i512 %arg) #2 {
  ret void
}

; CHECK: error: amdgpu kernarg preload offset exceeds the hardware limit
define amdgpu_kernel void @offset_too_large([2048 x i8] %unused, i32 %arg) #3 {
  ret void
}

; CHECK: error: unsupported argument in amdgpu kernarg preload range
define amdgpu_kernel void @unsupported_aggregate([2 x i32] %arg) #4 {
  ret void
}

; CHECK: error: amdgpu kernarg preload argument index exceeds the 32-bit limit
define amdgpu_kernel void @index_too_large(i32 %arg) #5 {
  ret void
}

attributes #0 = { "amdgpu-kernarg-preload-first-arg"="1" "amdgpu-kernarg-preload-last-arg"="0" }
attributes #1 = { "amdgpu-kernarg-preload-first-arg"="0" }
attributes #2 = { "amdgpu-kernarg-preload-first-arg"="0" "amdgpu-kernarg-preload-last-arg"="0" }
attributes #3 = { "amdgpu-kernarg-preload-first-arg"="1" "amdgpu-kernarg-preload-last-arg"="1" }
attributes #4 = { "amdgpu-kernarg-preload-first-arg"="0" "amdgpu-kernarg-preload-last-arg"="0" }
attributes #5 = { "amdgpu-kernarg-preload-first-arg"="4294967296" "amdgpu-kernarg-preload-last-arg"="4294967296" }
