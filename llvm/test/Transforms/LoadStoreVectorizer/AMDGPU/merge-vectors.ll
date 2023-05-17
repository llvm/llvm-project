; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7"

; CHECK-LABEL: @merge_v2i32_v2i32(
; CHECK: load <4 x i32>
; CHECK: store <4 x i32> zeroinitializer
define amdgpu_kernel void @merge_v2i32_v2i32(ptr addrspace(1) nocapture %a, ptr addrspace(1) nocapture readonly %b) #0 {
entry:
  %a.1 = getelementptr inbounds <2 x i32>, ptr addrspace(1) %a, i64 1
  %b.1 = getelementptr inbounds <2 x i32>, ptr addrspace(1) %b, i64 1

  %ld.c = load <2 x i32>, ptr addrspace(1) %b, align 4
  %ld.c.idx.1 = load <2 x i32>, ptr addrspace(1) %b.1, align 4

  store <2 x i32> zeroinitializer, ptr addrspace(1) %a, align 4
  store <2 x i32> zeroinitializer, ptr addrspace(1) %a.1, align 4

  ret void
}

; CHECK-LABEL: @merge_v1i32_v1i32(
; CHECK: load <2 x i32>
; CHECK: store <2 x i32> zeroinitializer
define amdgpu_kernel void @merge_v1i32_v1i32(ptr addrspace(1) nocapture %a, ptr addrspace(1) nocapture readonly %b) #0 {
entry:
  %a.1 = getelementptr inbounds <1 x i32>, ptr addrspace(1) %a, i64 1
  %b.1 = getelementptr inbounds <1 x i32>, ptr addrspace(1) %b, i64 1

  %ld.c = load <1 x i32>, ptr addrspace(1) %b, align 4
  %ld.c.idx.1 = load <1 x i32>, ptr addrspace(1) %b.1, align 4

  store <1 x i32> zeroinitializer, ptr addrspace(1) %a, align 4
  store <1 x i32> zeroinitializer, ptr addrspace(1) %a.1, align 4

  ret void
}

; CHECK-LABEL: @no_merge_v3i32_v3i32(
; CHECK: load <3 x i32>
; CHECK: load <3 x i32>
; CHECK: store <3 x i32> zeroinitializer
; CHECK: store <3 x i32> zeroinitializer
define amdgpu_kernel void @no_merge_v3i32_v3i32(ptr addrspace(1) nocapture %a, ptr addrspace(1) nocapture readonly %b) #0 {
entry:
  %a.1 = getelementptr inbounds <3 x i32>, ptr addrspace(1) %a, i64 1
  %b.1 = getelementptr inbounds <3 x i32>, ptr addrspace(1) %b, i64 1

  %ld.c = load <3 x i32>, ptr addrspace(1) %b, align 4
  %ld.c.idx.1 = load <3 x i32>, ptr addrspace(1) %b.1, align 4

  store <3 x i32> zeroinitializer, ptr addrspace(1) %a, align 4
  store <3 x i32> zeroinitializer, ptr addrspace(1) %a.1, align 4

  ret void
}

; CHECK-LABEL: @merge_v2i16_v2i16(
; CHECK: load <4 x i16>
; CHECK: store <4 x i16> zeroinitializer
define amdgpu_kernel void @merge_v2i16_v2i16(ptr addrspace(1) nocapture %a, ptr addrspace(1) nocapture readonly %b) #0 {
entry:
  %a.1 = getelementptr inbounds <2 x i16>, ptr addrspace(1) %a, i64 1
  %b.1 = getelementptr inbounds <2 x i16>, ptr addrspace(1) %b, i64 1

  %ld.c = load <2 x i16>, ptr addrspace(1) %b, align 4
  %ld.c.idx.1 = load <2 x i16>, ptr addrspace(1) %b.1, align 4

  store <2 x i16> zeroinitializer, ptr addrspace(1) %a, align 4
  store <2 x i16> zeroinitializer, ptr addrspace(1) %a.1, align 4

  ret void
}

; CHECK-LABEL: @merge_fat_ptrs(
; CHECK: load <4 x i16>
; CHECK: store <4 x i16> zeroinitializer
define amdgpu_kernel void @merge_fat_ptrs(ptr addrspace(7) nocapture %a, ptr addrspace(7) nocapture readonly %b) #0 {
entry:
  %a.1 = getelementptr inbounds <2 x i16>, ptr addrspace(7) %a, i32 1
  %b.1 = getelementptr inbounds <2 x i16>, ptr addrspace(7) %b, i32 1

  %ld.c = load <2 x i16>, ptr addrspace(7) %b, align 4
  %ld.c.idx.1 = load <2 x i16>, ptr addrspace(7) %b.1, align 4

  store <2 x i16> zeroinitializer, ptr addrspace(7) %a, align 4
  store <2 x i16> zeroinitializer, ptr addrspace(7) %a.1, align 4

  ret void
}

; Ideally this would be merged
; CHECK-LABEL: @merge_load_i32_v2i16(
; CHECK: load i32,
; CHECK: load <2 x i16>
define amdgpu_kernel void @merge_load_i32_v2i16(ptr addrspace(1) nocapture %a) #0 {
entry:
  %a.1 = getelementptr inbounds i32, ptr addrspace(1) %a, i32 1

  %ld.0 = load i32, ptr addrspace(1) %a
  %ld.1 = load <2 x i16>, ptr addrspace(1) %a.1

  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
