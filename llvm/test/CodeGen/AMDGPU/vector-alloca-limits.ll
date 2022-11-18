; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-promote-alloca,sroa,instcombine < %s | FileCheck -check-prefix=OPT %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-promote-alloca,sroa,instcombine -amdgpu-promote-alloca-to-vector-limit=32 < %s | FileCheck -check-prefix=LIMIT32 %s

target datalayout = "A5"

; OPT-LABEL: @alloca_8xi64_max1024(
; OPT-NOT: alloca
; OPT: <8 x i64>
; LIMIT32: alloca
; LIMIT32-NOT: <8 x i64>
define amdgpu_kernel void @alloca_8xi64_max1024(i64 addrspace(1)* %out, i32 %index) #0 {
entry:
  %tmp = alloca [8 x i64], addrspace(5)
  %x = getelementptr [8 x i64], [8 x i64] addrspace(5)* %tmp, i32 0, i32 0
  store i64 0, i64 addrspace(5)* %x
  %tmp1 = getelementptr [8 x i64], [8 x i64] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i64, i64 addrspace(5)* %tmp1
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_9xi64_max1024(
; OPT: alloca [9 x i64]
; OPT-NOT: <9 x i64>
; LIMIT32: alloca
; LIMIT32-NOT: <9 x i64>
define amdgpu_kernel void @alloca_9xi64_max1024(i64 addrspace(1)* %out, i32 %index) #0 {
entry:
  %tmp = alloca [9 x i64], addrspace(5)
  %x = getelementptr [9 x i64], [9 x i64] addrspace(5)* %tmp, i32 0, i32 0
  store i64 0, i64 addrspace(5)* %x
  %tmp1 = getelementptr [9 x i64], [9 x i64] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i64, i64 addrspace(5)* %tmp1
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_16xi64_max512(
; OPT-NOT: alloca
; OPT: <16 x i64>
; LIMIT32: alloca
; LIMIT32-NOT: <16 x i64>
define amdgpu_kernel void @alloca_16xi64_max512(i64 addrspace(1)* %out, i32 %index) #1 {
entry:
  %tmp = alloca [16 x i64], addrspace(5)
  %x = getelementptr [16 x i64], [16 x i64] addrspace(5)* %tmp, i32 0, i32 0
  store i64 0, i64 addrspace(5)* %x
  %tmp1 = getelementptr [16 x i64], [16 x i64] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i64, i64 addrspace(5)* %tmp1
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_17xi64_max512(
; OPT: alloca [17 x i64]
; OPT-NOT: <17 x i64>
; LIMIT32: alloca
; LIMIT32-NOT: <17 x i64>
define amdgpu_kernel void @alloca_17xi64_max512(i64 addrspace(1)* %out, i32 %index) #1 {
entry:
  %tmp = alloca [17 x i64], addrspace(5)
  %x = getelementptr [17 x i64], [17 x i64] addrspace(5)* %tmp, i32 0, i32 0
  store i64 0, i64 addrspace(5)* %x
  %tmp1 = getelementptr [17 x i64], [17 x i64] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i64, i64 addrspace(5)* %tmp1
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_9xi128_max512(
; OPT: alloca [9 x i128]
; OPT-NOT: <9 x i128>
; LIMIT32: alloca
; LIMIT32-NOT: <9 x i128>
define amdgpu_kernel void @alloca_9xi128_max512(i128 addrspace(1)* %out, i32 %index) #1 {
entry:
  %tmp = alloca [9 x i128], addrspace(5)
  %x = getelementptr [9 x i128], [9 x i128] addrspace(5)* %tmp, i32 0, i32 0
  store i128 0, i128 addrspace(5)* %x
  %tmp1 = getelementptr [9 x i128], [9 x i128] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i128, i128 addrspace(5)* %tmp1
  store i128 %tmp2, i128 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_9xi128_max256(
; OPT-NOT: alloca
; OPT: <9 x i128>
; LIMIT32: alloca
; LIMIT32-NOT: <9 x i128>
define amdgpu_kernel void @alloca_9xi128_max256(i128 addrspace(1)* %out, i32 %index) #2 {
entry:
  %tmp = alloca [9 x i128], addrspace(5)
  %x = getelementptr [9 x i128], [9 x i128] addrspace(5)* %tmp, i32 0, i32 0
  store i128 0, i128 addrspace(5)* %x
  %tmp1 = getelementptr [9 x i128], [9 x i128] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i128, i128 addrspace(5)* %tmp1
  store i128 %tmp2, i128 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_16xi128_max256(
; OPT-NOT: alloca
; OPT: <16 x i128>
; LIMIT32: alloca
; LIMIT32-NOT: <16 x i128>
define amdgpu_kernel void @alloca_16xi128_max256(i128 addrspace(1)* %out, i32 %index) #2 {
entry:
  %tmp = alloca [16 x i128], addrspace(5)
  %x = getelementptr [16 x i128], [16 x i128] addrspace(5)* %tmp, i32 0, i32 0
  store i128 0, i128 addrspace(5)* %x
  %tmp1 = getelementptr [16 x i128], [16 x i128] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i128, i128 addrspace(5)* %tmp1
  store i128 %tmp2, i128 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_9xi256_max256(
; OPT: alloca [9 x i256]
; OPT-NOT: <9 x i256>
; LIMIT32: alloca
; LIMIT32-NOT: <9 x i256>
define amdgpu_kernel void @alloca_9xi256_max256(i256 addrspace(1)* %out, i32 %index) #2 {
entry:
  %tmp = alloca [9 x i256], addrspace(5)
  %x = getelementptr [9 x i256], [9 x i256] addrspace(5)* %tmp, i32 0, i32 0
  store i256 0, i256 addrspace(5)* %x
  %tmp1 = getelementptr [9 x i256], [9 x i256] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i256, i256 addrspace(5)* %tmp1
  store i256 %tmp2, i256 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @alloca_9xi64_max256(
; OPT-NOT: alloca
; OPT: <9 x i64>
; LIMIT32: alloca
; LIMIT32-NOT: <9 x i64>
define amdgpu_kernel void @alloca_9xi64_max256(i64 addrspace(1)* %out, i32 %index) #2 {
entry:
  %tmp = alloca [9 x i64], addrspace(5)
  %x = getelementptr [9 x i64], [9 x i64] addrspace(5)* %tmp, i32 0, i32 0
  store i64 0, i64 addrspace(5)* %x
  %tmp1 = getelementptr [9 x i64], [9 x i64] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i64, i64 addrspace(5)* %tmp1
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @func_alloca_9xi64_max256(
; OPT: alloca
; OPT-NOT: <9 x i64>
; LIMIT32: alloca
; LIMIT32-NOT: <9 x i64>
define void @func_alloca_9xi64_max256(i64 addrspace(1)* %out, i32 %index) #2 {
entry:
  %tmp = alloca [9 x i64], addrspace(5)
  %x = getelementptr [9 x i64], [9 x i64] addrspace(5)* %tmp, i32 0, i32 0
  store i64 0, i64 addrspace(5)* %x
  %tmp1 = getelementptr [9 x i64], [9 x i64] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i64, i64 addrspace(5)* %tmp1
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,512" }
attributes #2 = { "amdgpu-flat-work-group-size"="1,256" }
