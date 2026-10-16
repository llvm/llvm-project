; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call <4 x i32> @llvm.amdgcn.av.load.b128.p1({{.*}})
define <4 x i32> @av_global_load_b128_bad_metadata(ptr addrspace(1) %addr) {
  %data = call <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1) %addr, metadata i32 0)
  ret <4 x i32> %data
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.av.store.b128.p1({{.*}})
define void @av_global_store_b128_bad_metadata(ptr addrspace(1) %addr, <4 x i32> %data) {
  call void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1) %addr, <4 x i32> %data, metadata i32 0)
  ret void
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call <4 x i32> @llvm.amdgcn.av.load.b128.p0({{.*}})
define <4 x i32> @av_flat_load_b128_bad_metadata(ptr %addr) {
  %data = call <4 x i32> @llvm.amdgcn.av.load.b128.p0(ptr %addr, metadata i32 0)
  ret <4 x i32> %data
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.av.store.b128.p0({{.*}})
define void @av_flat_store_b128_bad_metadata(ptr %addr, <4 x i32> %data) {
  call void @llvm.amdgcn.av.store.b128.p0(ptr %addr, <4 x i32> %data, metadata i32 0)
  ret void
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call i8 @llvm.amdgcn.av.load.b8.p1({{.*}})
define i8 @av_global_load_b8_bad_metadata(ptr addrspace(1) %addr) {
  %data = call i8 @llvm.amdgcn.av.load.b8.p1(ptr addrspace(1) %addr, metadata i32 0)
  ret i8 %data
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.av.store.b8.p1({{.*}})
define void @av_global_store_b8_bad_metadata(ptr addrspace(1) %addr, i8 %data) {
  call void @llvm.amdgcn.av.store.b8.p1(ptr addrspace(1) %addr, i8 %data, metadata i32 0)
  ret void
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call i16 @llvm.amdgcn.av.load.b16.p1({{.*}})
define i16 @av_global_load_b16_bad_metadata(ptr addrspace(1) %addr) {
  %data = call i16 @llvm.amdgcn.av.load.b16.p1(ptr addrspace(1) %addr, metadata i32 0)
  ret i16 %data
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.av.store.b16.p1({{.*}})
define void @av_global_store_b16_bad_metadata(ptr addrspace(1) %addr, i16 %data) {
  call void @llvm.amdgcn.av.store.b16.p1(ptr addrspace(1) %addr, i16 %data, metadata i32 0)
  ret void
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call i32 @llvm.amdgcn.av.load.b32.p1({{.*}})
define i32 @av_global_load_b32_bad_metadata(ptr addrspace(1) %addr) {
  %data = call i32 @llvm.amdgcn.av.load.b32.p1(ptr addrspace(1) %addr, metadata i32 0)
  ret i32 %data
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.av.store.b32.p1({{.*}})
define void @av_global_store_b32_bad_metadata(ptr addrspace(1) %addr, i32 %data) {
  call void @llvm.amdgcn.av.store.b32.p1(ptr addrspace(1) %addr, i32 %data, metadata i32 0)
  ret void
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call <2 x i32> @llvm.amdgcn.av.load.b64.p1({{.*}})
define <2 x i32> @av_global_load_b64_bad_metadata(ptr addrspace(1) %addr) {
  %data = call <2 x i32> @llvm.amdgcn.av.load.b64.p1(ptr addrspace(1) %addr, metadata i32 0)
  ret <2 x i32> %data
}

; CHECK: the last argument to av load/store intrinsics must be a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.av.store.b64.p1({{.*}})
define void @av_global_store_b64_bad_metadata(ptr addrspace(1) %addr, <2 x i32> %data) {
  call void @llvm.amdgcn.av.store.b64.p1(ptr addrspace(1) %addr, <2 x i32> %data, metadata i32 0)
  ret void
}
