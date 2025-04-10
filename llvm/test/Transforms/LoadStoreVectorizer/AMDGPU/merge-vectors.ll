; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -mattr=+relaxed-buffer-oob-mode -S -o - %s | FileCheck --check-prefixes=CHECK,CHECK-OOB-RELAXED %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck --check-prefixes=CHECK,CHECK-OOB-STRICT %s

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

; CHECK-LABEL: @merge_load_i32_v2i16(
; CHECK: load <2 x i32>
; CHECK: extractelement <2 x i32> %0, i32 0
; CHECK: extractelement <2 x i32> %0, i32 1
define amdgpu_kernel void @merge_load_i32_v2i16(ptr addrspace(1) nocapture %a) #0 {
entry:
  %a.1 = getelementptr inbounds i32, ptr addrspace(1) %a, i32 1

  %ld.0 = load i32, ptr addrspace(1) %a
  %ld.1 = load <2 x i16>, ptr addrspace(1) %a.1

  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

; CHECK-LABEL: @merge_i32_2i16_float_4i8(
; CHECK: load <4 x i32>
; CHECK: store <2 x i32>
; CHECK: store <2 x i32>
define void @merge_i32_2i16_float_4i8(ptr addrspace(1) %ptr1, ptr addrspace(2) %ptr2) {
  %gep1 = getelementptr inbounds i32, ptr addrspace(1) %ptr1, i64 0
  %load1 = load i32, ptr addrspace(1) %gep1, align 4
  %gep2 = getelementptr inbounds <2 x i16>, ptr addrspace(1) %ptr1, i64 1
  %load2 = load <2 x i16>, ptr addrspace(1) %gep2, align 4
  %gep3 = getelementptr inbounds float, ptr addrspace(1) %ptr1, i64 2
  %load3 = load float, ptr addrspace(1) %gep3, align 4
  %gep4 = getelementptr inbounds <4 x i8>, ptr addrspace(1) %ptr1, i64 3
  %load4 = load <4 x i8>, ptr addrspace(1) %gep4, align 4
  %store.gep1 = getelementptr inbounds i32, ptr addrspace(2) %ptr2, i64 0
  store i32 %load1, ptr addrspace(2) %store.gep1, align 4
  %store.gep2 = getelementptr inbounds <2 x i16>, ptr addrspace(2) %ptr2, i64 1
  store <2 x i16> %load2, ptr addrspace(2) %store.gep2, align 4
  %store.gep3 = getelementptr inbounds float, ptr addrspace(2) %ptr2, i64 2
  store float %load3, ptr addrspace(2) %store.gep3, align 4
  %store.gep4 = getelementptr inbounds <4 x i8>, ptr addrspace(2) %ptr2, i64 3
  store <4 x i8> %load4, ptr addrspace(2) %store.gep4, align 4
  ret void
}

; CHECK-LABEL: @merge_fp_type(
; CHECK: load <2 x float>
; CHECK: bitcast float {{.*}} to <2 x half>
define void @merge_fp_type(ptr addrspace(1) %ptr1, ptr addrspace(2) %ptr2) {
  %gep1 = getelementptr inbounds float, ptr addrspace(1) %ptr1, i64 0
  %load1 = load float, ptr addrspace(1) %gep1, align 4
  %gep2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %ptr1, i64 1
  %load2 = load <2 x half>, ptr addrspace(1) %gep2, align 4
  ret void
}
