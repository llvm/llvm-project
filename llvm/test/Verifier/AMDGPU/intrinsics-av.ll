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
