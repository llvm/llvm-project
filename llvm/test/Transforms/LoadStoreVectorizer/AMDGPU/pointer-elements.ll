; RUN: opt -mtriple=amdgcn-amd-amdhsa -basicaa -load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

declare i32 @llvm.amdgcn.workitem.id.x() #1

; CHECK-LABEL: @merge_v2p1i8(
; CHECK: load <2 x i64>
; CHECK: inttoptr i64 %{{[^ ]+}} to i8 addrspace(1)*
; CHECK: inttoptr i64 %{{[^ ]+}} to i8 addrspace(1)*
; CHECK: store <2 x i64> zeroinitializer
define amdgpu_kernel void @merge_v2p1i8(i8 addrspace(1)* addrspace(1)* nocapture %a, i8 addrspace(1)* addrspace(1)* nocapture readonly %b) #0 {
entry:
  %a.1 = getelementptr inbounds i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %a, i64 1
  %b.1 = getelementptr inbounds i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %b, i64 1

  %ld.c = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %b, align 4
  %ld.c.idx.1 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %b.1, align 4

  store i8 addrspace(1)* null, i8 addrspace(1)* addrspace(1)* %a, align 4
  store i8 addrspace(1)* null, i8 addrspace(1)* addrspace(1)* %a.1, align 4

  ret void
}

; CHECK-LABEL: @merge_v2p3i8(
; CHECK: load <2 x i32>
; CHECK: inttoptr i32 %{{[^ ]+}} to i8 addrspace(3)*
; CHECK: inttoptr i32 %{{[^ ]+}} to i8 addrspace(3)*
; CHECK: store <2 x i32> zeroinitializer
define amdgpu_kernel void @merge_v2p3i8(i8 addrspace(3)* addrspace(3)* nocapture %a, i8 addrspace(3)* addrspace(3)* nocapture readonly %b) #0 {
entry:
  %a.1 = getelementptr inbounds i8 addrspace(3)*, i8 addrspace(3)* addrspace(3)* %a, i64 1
  %b.1 = getelementptr inbounds i8 addrspace(3)*, i8 addrspace(3)* addrspace(3)* %b, i64 1

  %ld.c = load i8 addrspace(3)*, i8 addrspace(3)* addrspace(3)* %b, align 4
  %ld.c.idx.1 = load i8 addrspace(3)*, i8 addrspace(3)* addrspace(3)* %b.1, align 4

  store i8 addrspace(3)* null, i8 addrspace(3)* addrspace(3)* %a, align 4
  store i8 addrspace(3)* null, i8 addrspace(3)* addrspace(3)* %a.1, align 4

  ret void
}

; CHECK-LABEL: @merge_load_i64_ptr64(
; CHECK: load <2 x i64>
; CHECK: [[ELT1:%[^ ]+]] = extractelement <2 x i64> %{{[^ ]+}}, i32 1
; CHECK: inttoptr i64 [[ELT1]] to i8 addrspace(1)*
define amdgpu_kernel void @merge_load_i64_ptr64(i64 addrspace(1)* nocapture %a) #0 {
entry:
  %a.1 = getelementptr inbounds i64, i64 addrspace(1)* %a, i64 1
  %a.1.cast = bitcast i64 addrspace(1)* %a.1 to i8 addrspace(1)* addrspace(1)*

  %ld.0 = load i64, i64 addrspace(1)* %a
  %ld.1 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %a.1.cast

  ret void
}

; CHECK-LABEL: @merge_load_ptr64_i64(
; CHECK: load <2 x i64>
; CHECK: [[ELT0:%[^ ]+]] = extractelement <2 x i64> %{{[^ ]+}}, i32 0
; CHECK: inttoptr i64 [[ELT0]] to i8 addrspace(1)*
define amdgpu_kernel void @merge_load_ptr64_i64(i64 addrspace(1)* nocapture %a) #0 {
entry:
  %a.cast = bitcast i64 addrspace(1)* %a to i8 addrspace(1)* addrspace(1)*
  %a.1 =  getelementptr inbounds i64, i64 addrspace(1)* %a, i64 1

  %ld.0 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %a.cast
  %ld.1 = load i64, i64 addrspace(1)* %a.1

  ret void
}

; CHECK-LABEL: @merge_store_ptr64_i64(
; CHECK: [[ELT0:%[^ ]+]] = ptrtoint i8 addrspace(1)* %ptr0 to i64
; CHECK: insertelement <2 x i64> undef, i64 [[ELT0]], i32 0
; CHECK: store <2 x i64>
define amdgpu_kernel void @merge_store_ptr64_i64(i64 addrspace(1)* nocapture %a, i8 addrspace(1)* %ptr0, i64 %val1) #0 {
entry:
  %a.cast = bitcast i64 addrspace(1)* %a to i8 addrspace(1)* addrspace(1)*
  %a.1 = getelementptr inbounds i64, i64 addrspace(1)* %a, i64 1


  store i8 addrspace(1)* %ptr0, i8 addrspace(1)* addrspace(1)* %a.cast
  store i64 %val1, i64 addrspace(1)* %a.1

  ret void
}

; CHECK-LABEL: @merge_store_i64_ptr64(
; CHECK: [[ELT1:%[^ ]+]] = ptrtoint i8 addrspace(1)* %ptr1 to i64
; CHECK: insertelement <2 x i64> %{{[^ ]+}}, i64 [[ELT1]], i32 1
; CHECK: store <2 x i64>
define amdgpu_kernel void @merge_store_i64_ptr64(i8 addrspace(1)* addrspace(1)* nocapture %a, i64 %val0, i8 addrspace(1)* %ptr1) #0 {
entry:
  %a.1 = getelementptr inbounds i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %a, i64 1
  %a.cast = bitcast i8 addrspace(1)* addrspace(1)* %a to i64 addrspace(1)*

  store i64 %val0, i64 addrspace(1)* %a.cast
  store i8 addrspace(1)* %ptr1, i8 addrspace(1)* addrspace(1)* %a.1

  ret void
}

; CHECK-LABEL: @merge_load_i32_ptr32(
; CHECK: load <2 x i32>
; CHECK: [[ELT1:%[^ ]+]] = extractelement <2 x i32> %{{[^ ]+}}, i32 1
; CHECK: inttoptr i32 [[ELT1]] to i8 addrspace(3)*
define amdgpu_kernel void @merge_load_i32_ptr32(i32 addrspace(3)* nocapture %a) #0 {
entry:
  %a.1 = getelementptr inbounds i32, i32 addrspace(3)* %a, i32 1
  %a.1.cast = bitcast i32 addrspace(3)* %a.1 to i8 addrspace(3)* addrspace(3)*

  %ld.0 = load i32, i32 addrspace(3)* %a
  %ld.1 = load i8 addrspace(3)*, i8 addrspace(3)* addrspace(3)* %a.1.cast

  ret void
}

; CHECK-LABEL: @merge_load_ptr32_i32(
; CHECK: load <2 x i32>
; CHECK: [[ELT0:%[^ ]+]] = extractelement <2 x i32> %{{[^ ]+}}, i32 0
; CHECK: inttoptr i32 [[ELT0]] to i8 addrspace(3)*
define amdgpu_kernel void @merge_load_ptr32_i32(i32 addrspace(3)* nocapture %a) #0 {
entry:
  %a.cast = bitcast i32 addrspace(3)* %a to i8 addrspace(3)* addrspace(3)*
  %a.1 = getelementptr inbounds i32, i32 addrspace(3)* %a, i32 1

  %ld.0 = load i8 addrspace(3)*, i8 addrspace(3)* addrspace(3)* %a.cast
  %ld.1 = load i32, i32 addrspace(3)* %a.1

  ret void
}

; CHECK-LABEL: @merge_store_ptr32_i32(
; CHECK: [[ELT0:%[^ ]+]] = ptrtoint i8 addrspace(3)* %ptr0 to i32
; CHECK: insertelement <2 x i32> undef, i32 [[ELT0]], i32 0
; CHECK: store <2 x i32>
define amdgpu_kernel void @merge_store_ptr32_i32(i32 addrspace(3)* nocapture %a, i8 addrspace(3)* %ptr0, i32 %val1) #0 {
entry:
  %a.cast = bitcast i32 addrspace(3)* %a to i8 addrspace(3)* addrspace(3)*
  %a.1 = getelementptr inbounds i32, i32 addrspace(3)* %a, i32 1

  store i8 addrspace(3)* %ptr0, i8 addrspace(3)* addrspace(3)* %a.cast
  store i32 %val1, i32 addrspace(3)* %a.1

  ret void
}

; CHECK-LABEL: @merge_store_i32_ptr32(
; CHECK: [[ELT1:%[^ ]+]] = ptrtoint i8 addrspace(3)* %ptr1 to i32
; CHECK: insertelement <2 x i32> %{{[^ ]+}}, i32 [[ELT1]], i32 1
; CHECK: store <2 x i32>
define amdgpu_kernel void @merge_store_i32_ptr32(i8 addrspace(3)* addrspace(3)* nocapture %a, i32 %val0, i8 addrspace(3)* %ptr1) #0 {
entry:
  %a.1 = getelementptr inbounds i8 addrspace(3)*, i8 addrspace(3)* addrspace(3)* %a, i32 1
  %a.cast = bitcast i8 addrspace(3)* addrspace(3)* %a to i32 addrspace(3)*

  store i32 %val0, i32 addrspace(3)* %a.cast
  store i8 addrspace(3)* %ptr1, i8 addrspace(3)* addrspace(3)* %a.1

  ret void
}

; CHECK-LABEL: @no_merge_store_ptr32_i64(
; CHECK: store i8 addrspace(3)*
; CHECK: store i64
define amdgpu_kernel void @no_merge_store_ptr32_i64(i64 addrspace(1)* nocapture %a, i8 addrspace(3)* %ptr0, i64 %val1) #0 {
entry:
  %a.cast = bitcast i64 addrspace(1)* %a to i8 addrspace(3)* addrspace(1)*
  %a.1 = getelementptr inbounds i64, i64 addrspace(1)* %a, i64 1


  store i8 addrspace(3)* %ptr0, i8 addrspace(3)* addrspace(1)* %a.cast
  store i64 %val1, i64 addrspace(1)* %a.1

  ret void
}

; CHECK-LABEL: @no_merge_store_i64_ptr32(
; CHECK: store i64
; CHECK: store i8 addrspace(3)*
define amdgpu_kernel void @no_merge_store_i64_ptr32(i8 addrspace(3)* addrspace(1)* nocapture %a, i64 %val0, i8 addrspace(3)* %ptr1) #0 {
entry:
  %a.1 =  getelementptr inbounds i8 addrspace(3)*, i8 addrspace(3)* addrspace(1)* %a, i64 1
  %a.cast = bitcast i8 addrspace(3)* addrspace(1)* %a to i64 addrspace(1)*

  store i64 %val0, i64 addrspace(1)* %a.cast
  store i8 addrspace(3)* %ptr1, i8 addrspace(3)* addrspace(1)* %a.1

  ret void
}

; CHECK-LABEL: @no_merge_load_i64_ptr32(
; CHECK: load i64,
; CHECK: load i8 addrspace(3)*,
define amdgpu_kernel void @no_merge_load_i64_ptr32(i64 addrspace(1)* nocapture %a) #0 {
entry:
  %a.1 = getelementptr inbounds i64, i64 addrspace(1)* %a, i64 1
  %a.1.cast = bitcast i64 addrspace(1)* %a.1 to i8 addrspace(3)* addrspace(1)*

  %ld.0 = load i64, i64 addrspace(1)* %a
  %ld.1 = load i8 addrspace(3)*, i8 addrspace(3)* addrspace(1)* %a.1.cast

  ret void
}

; CHECK-LABEL: @no_merge_load_ptr32_i64(
; CHECK: load i8 addrspace(3)*,
; CHECK: load i64,
define amdgpu_kernel void @no_merge_load_ptr32_i64(i64 addrspace(1)* nocapture %a) #0 {
entry:
  %a.cast = bitcast i64 addrspace(1)* %a to i8 addrspace(3)* addrspace(1)*
  %a.1 =  getelementptr inbounds i64, i64 addrspace(1)* %a, i64 1

  %ld.0 = load i8 addrspace(3)*, i8 addrspace(3)* addrspace(1)* %a.cast
  %ld.1 = load i64, i64 addrspace(1)* %a.1

  ret void
}

; XXX - This isn't merged for some reason
; CHECK-LABEL: @merge_v2p1i8_v2p1i8(
; CHECK: load <2 x i8 addrspace(1)*>
; CHECK: load <2 x i8 addrspace(1)*>
; CHECK: store <2 x i8 addrspace(1)*>
; CHECK: store <2 x i8 addrspace(1)*>
define amdgpu_kernel void @merge_v2p1i8_v2p1i8(<2 x i8 addrspace(1)*> addrspace(1)* nocapture noalias %a, <2 x i8 addrspace(1)*> addrspace(1)* nocapture readonly noalias %b) #0 {
entry:
  %a.1 = getelementptr inbounds <2 x i8 addrspace(1)*>, <2 x i8 addrspace(1)*> addrspace(1)* %a, i64 1
  %b.1 = getelementptr inbounds <2 x i8 addrspace(1)*>, <2 x i8 addrspace(1)*> addrspace(1)* %b, i64 1

  %ld.c = load <2 x i8 addrspace(1)*>, <2 x i8 addrspace(1)*> addrspace(1)* %b, align 4
  %ld.c.idx.1 = load <2 x i8 addrspace(1)*>, <2 x i8 addrspace(1)*> addrspace(1)* %b.1, align 4

  store <2 x i8 addrspace(1)*> zeroinitializer, <2 x i8 addrspace(1)*> addrspace(1)* %a, align 4
  store <2 x i8 addrspace(1)*> zeroinitializer, <2 x i8 addrspace(1)*> addrspace(1)* %a.1, align 4
  ret void
}

; CHECK-LABEL: @merge_load_ptr64_f64(
; CHECK: load <2 x i64>
; CHECK: [[ELT0:%[^ ]+]] = extractelement <2 x i64> %{{[^ ]+}}, i32 0
; CHECK: [[ELT0_INT:%[^ ]+]] = inttoptr i64 [[ELT0]] to i8 addrspace(1)*
; CHECK: [[ELT1_INT:%[^ ]+]] = extractelement <2 x i64> %{{[^ ]+}}, i32 1
; CHECK: bitcast i64 [[ELT1_INT]] to double
define amdgpu_kernel void @merge_load_ptr64_f64(double addrspace(1)* nocapture %a) #0 {
entry:
  %a.cast = bitcast double addrspace(1)* %a to i8 addrspace(1)* addrspace(1)*
  %a.1 =  getelementptr inbounds double, double addrspace(1)* %a, i64 1

  %ld.0 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %a.cast
  %ld.1 = load double, double addrspace(1)* %a.1

  ret void
}

; CHECK-LABEL: @merge_load_f64_ptr64(
; CHECK: load <2 x i64>
; CHECK: [[ELT0:%[^ ]+]] = extractelement <2 x i64> %{{[^ ]+}}, i32 0
; CHECK: bitcast i64 [[ELT0]] to double
; CHECK: [[ELT1:%[^ ]+]] = extractelement <2 x i64> %{{[^ ]+}}, i32 1
; CHECK: inttoptr i64 [[ELT1]] to i8 addrspace(1)*
define amdgpu_kernel void @merge_load_f64_ptr64(double addrspace(1)* nocapture %a) #0 {
entry:
  %a.1 = getelementptr inbounds double, double addrspace(1)* %a, i64 1
  %a.1.cast = bitcast double addrspace(1)* %a.1 to i8 addrspace(1)* addrspace(1)*

  %ld.0 = load double, double addrspace(1)* %a
  %ld.1 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %a.1.cast

  ret void
}

; CHECK-LABEL: @merge_store_ptr64_f64(
; CHECK: [[ELT0_INT:%[^ ]+]] = ptrtoint i8 addrspace(1)* %ptr0 to i64
; CHECK: insertelement <2 x i64> undef, i64 [[ELT0_INT]], i32 0
; CHECK: [[ELT1_INT:%[^ ]+]] = bitcast double %val1 to i64
; CHECK: insertelement <2 x i64> %{{[^ ]+}}, i64 [[ELT1_INT]], i32 1
; CHECK: store <2 x i64>
define amdgpu_kernel void @merge_store_ptr64_f64(double addrspace(1)* nocapture %a, i8 addrspace(1)* %ptr0, double %val1) #0 {
entry:
  %a.cast = bitcast double addrspace(1)* %a to i8 addrspace(1)* addrspace(1)*
  %a.1 = getelementptr inbounds double, double addrspace(1)* %a, i64 1

  store i8 addrspace(1)* %ptr0, i8 addrspace(1)* addrspace(1)* %a.cast
  store double %val1, double addrspace(1)* %a.1

  ret void
}

; CHECK-LABEL: @merge_store_f64_ptr64(
; CHECK: [[ELT0_INT:%[^ ]+]] = bitcast double %val0 to i64
; CHECK: insertelement <2 x i64> undef, i64 [[ELT0_INT]], i32 0
; CHECK: [[ELT1_INT:%[^ ]+]] = ptrtoint i8 addrspace(1)* %ptr1 to i64
; CHECK: insertelement <2 x i64> %{{[^ ]+}}, i64 [[ELT1_INT]], i32 1
; CHECK: store <2 x i64>
define amdgpu_kernel void @merge_store_f64_ptr64(i8 addrspace(1)* addrspace(1)* nocapture %a, double %val0, i8 addrspace(1)* %ptr1) #0 {
entry:
  %a.1 = getelementptr inbounds i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %a, i64 1
  %a.cast = bitcast i8 addrspace(1)* addrspace(1)* %a to double addrspace(1)*

  store double %val0, double addrspace(1)* %a.cast
  store i8 addrspace(1)* %ptr1, i8 addrspace(1)* addrspace(1)* %a.1

  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
