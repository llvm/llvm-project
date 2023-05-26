; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Checks that we don't merge loads/stores of types smaller than one
; byte, or vectors with elements smaller than one byte.

%struct.foo = type { i32, i8 }

declare void @use_i1(i1)
declare void @use_i2(i2)
declare void @use_i8(i8)
declare void @use_foo(%struct.foo)
declare void @use_v2i2(<2 x i2>)
declare void @use_v4i2(<4 x i2>)
declare void @use_v2i9(<2 x i9>)

; CHECK-LABEL: @merge_store_2_constants_i1(
; CHECK: store i1
; CHECK: store i1
define amdgpu_kernel void @merge_store_2_constants_i1(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i1, ptr addrspace(1) %out, i32 1
  store i1 true, ptr addrspace(1) %out.gep.1
  store i1 false, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @merge_store_2_constants_i2(
; CHECK: store i2 1
; CHECK: store i2 -1
define amdgpu_kernel void @merge_store_2_constants_i2(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i2, ptr addrspace(1) %out, i32 1
  store i2 1, ptr addrspace(1) %out.gep.1
  store i2 -1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @merge_different_store_sizes_i1_i8(
; CHECK: store i1 true
; CHECK: store i8 123
define amdgpu_kernel void @merge_different_store_sizes_i1_i8(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i32 1
  store i1 true, ptr addrspace(1) %out
  store i8 123, ptr addrspace(1) %out.gep.1
  ret void
}

; CHECK-LABEL: @merge_different_store_sizes_i8_i1(
; CHECK: store i8 123
; CHECK: store i1 true
define amdgpu_kernel void @merge_different_store_sizes_i8_i1(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i32 1
  store i8 123, ptr addrspace(1) %out.gep.1
  store i1 true, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @merge_store_2_constant_structs(
; CHECK: store %struct.foo
; CHECK: store %struct.foo
define amdgpu_kernel void @merge_store_2_constant_structs(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr %struct.foo, ptr addrspace(1) %out, i32 1
  store %struct.foo { i32 12, i8 3 }, ptr addrspace(1) %out.gep.1
  store %struct.foo { i32 92, i8 9 }, ptr addrspace(1) %out
  ret void
}

; sub-byte element size
; CHECK-LABEL: @merge_store_2_constants_v2i2(
; CHECK: store <2 x i2>
; CHECK: store <2 x i2>
define amdgpu_kernel void @merge_store_2_constants_v2i2(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr <2 x i2>, ptr addrspace(1) %out, i32 1
  store <2 x i2> <i2 1, i2 -1>, ptr addrspace(1) %out.gep.1
  store <2 x i2> <i2 -1, i2 1>, ptr addrspace(1) %out
  ret void
}

; sub-byte element size but byte size

; CHECK-LABEL: @merge_store_2_constants_v4i2(
; CHECK: store <4 x i2>
; CHECK: store <4 x i2>
define amdgpu_kernel void @merge_store_2_constants_v4i2(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr <4 x i2>, ptr addrspace(1) %out, i32 1
  store <4 x i2> <i2 1, i2 -1, i2 1, i2 -1>, ptr addrspace(1) %out.gep.1
  store <4 x i2> <i2 -1, i2 1, i2 -1, i2 1>, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_i1(
; CHECK: load i1
; CHECK: load i1
define amdgpu_kernel void @merge_load_2_constants_i1(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i1, ptr addrspace(1) %out, i32 1
  %x = load i1, ptr addrspace(1) %out.gep.1
  %y = load i1, ptr addrspace(1) %out
  call void @use_i1(i1 %x)
  call void @use_i1(i1 %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_i2(
; CHECK: load i2
; CHECK: load i2
define amdgpu_kernel void @merge_load_2_constants_i2(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i2, ptr addrspace(1) %out, i32 1
  %x = load i2, ptr addrspace(1) %out.gep.1
  %y = load i2, ptr addrspace(1) %out
  call void @use_i2(i2 %x)
  call void @use_i2(i2 %y)
  ret void
}

; CHECK-LABEL: @merge_different_load_sizes_i1_i8(
; CHECK: load i1
; CHECK: load i8
define amdgpu_kernel void @merge_different_load_sizes_i1_i8(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i32 1
  %x = load i1, ptr addrspace(1) %out
  %y = load i8, ptr addrspace(1) %out.gep.1
  call void @use_i1(i1 %x)
  call void @use_i8(i8 %y)
  ret void
}

; CHECK-LABEL: @merge_different_load_sizes_i8_i1(
; CHECK: load i8
; CHECK: load i1
define amdgpu_kernel void @merge_different_load_sizes_i8_i1(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i32 1
  %x = load i8, ptr addrspace(1) %out.gep.1
  %y = load i1, ptr addrspace(1) %out
  call void @use_i8(i8 %x)
  call void @use_i1(i1 %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constant_structs(
; CHECK: load %struct.foo
; CHECK: load %struct.foo
define amdgpu_kernel void @merge_load_2_constant_structs(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr %struct.foo, ptr addrspace(1) %out, i32 1
  %x = load %struct.foo, ptr addrspace(1) %out.gep.1
  %y = load %struct.foo, ptr addrspace(1) %out
  call void @use_foo(%struct.foo %x)
  call void @use_foo(%struct.foo %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_v2i2(
; CHECK: load <2 x i2>
; CHECK: load <2 x i2>
define amdgpu_kernel void @merge_load_2_constants_v2i2(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr <2 x i2>, ptr addrspace(1) %out, i32 1
  %x = load <2 x i2>, ptr addrspace(1) %out.gep.1
  %y = load <2 x i2>, ptr addrspace(1) %out
  call void @use_v2i2(<2 x i2> %x)
  call void @use_v2i2(<2 x i2> %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_v4i2(
; CHECK: load <4 x i2>
; CHECK: load <4 x i2>
define amdgpu_kernel void @merge_load_2_constants_v4i2(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr <4 x i2>, ptr addrspace(1) %out, i32 1
  %x = load <4 x i2>, ptr addrspace(1) %out.gep.1
  %y = load <4 x i2>, ptr addrspace(1) %out
  call void @use_v4i2(<4 x i2> %x)
  call void @use_v4i2(<4 x i2> %y)
  ret void
}

; CHECK-LABEL: @merge_store_2_constants_i9(
; CHECK: store i9 3
; CHECK: store i9 -5
define amdgpu_kernel void @merge_store_2_constants_i9(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i9, ptr addrspace(1) %out, i32 1
  store i9 3, ptr addrspace(1) %out.gep.1
  store i9 -5, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_v2i9(
; CHECK: load <2 x i9>
; CHECK: load <2 x i9>
define amdgpu_kernel void @merge_load_2_constants_v2i9(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr <2 x i9>, ptr addrspace(1) %out, i32 1
  %x = load <2 x i9>, ptr addrspace(1) %out.gep.1
  %y = load <2 x i9>, ptr addrspace(1) %out
  call void @use_v2i9(<2 x i9> %x)
  call void @use_v2i9(<2 x i9> %y)
  ret void
}

attributes #0 = { nounwind }
