; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -mattr=+relaxed-buffer-oob-mode -S -o - %s | FileCheck --check-prefixes=CHECK,CHECK-OOB-RELAXED %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck --check-prefixes=CHECK,CHECK-OOB-STRICT %s

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

; CHECK-OOB-RELAXED-LABEL: @merge_fat_ptrs(
; CHECK-OOB-RELAXED: load <4 x i16>
; CHECK-OOB-RELAXED: store <4 x i16> zeroinitializer
; CHECK-OOB-STRICT-LABEL: @merge_fat_ptrs(
; CHECK-OOB-STRICT: load <2 x i16>
; CHECK-OOB-STRICT: load <2 x i16>
; CHECK-OOB-STRICT: store <2 x i16> zeroinitializer
; CHECK-OOB-STRICT: store <2 x i16> zeroinitializer
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
