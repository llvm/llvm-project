; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; ---------- i32 metadata ------------------------------------------------------
; CHECK: global load/store intrinsics require that the last argument is a metadata string
; CHECK-NEXT: call <4 x i32> @llvm.amdgcn.global.load.b128({{.*}})
; CHECK-NEXT: metadata i32 1
define <4 x i32> @global_load_b128_00(ptr addrspace(1) %addr) {
entry:
  %data = call <4 x i32> @llvm.amdgcn.global.load.b128(ptr addrspace(1) %addr, metadata !3)
  ret <4 x i32> %data
}

; CHECK: global load/store intrinsics require that the last argument is a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.global.store.b128({{.*}})
; CHECK-NEXT: metadata i32 1
define void @global_store_b128_00(ptr addrspace(1) %addr, <4 x i32> %data) {
entry:
  call void @llvm.amdgcn.global.store.b128(ptr addrspace(1) %addr, <4 x i32> %data, metadata !3)
  ret void
}

; ---------- non-tuple metadata ------------------------------------------------
; CHECK:      global load/store intrinsics require that the last argument is a metadata string
; CHECK-NEXT: call <4 x i32> @llvm.amdgcn.global.load.b128({{.*}})
; CHECK-NEXT: metadata !0
define <4 x i32> @global_load_b128_01(ptr addrspace(1) %addr) {
entry:
  %data = call <4 x i32> @llvm.amdgcn.global.load.b128(ptr addrspace(1) %addr, metadata !0)
  ret <4 x i32> %data
}

; CHECK:      global load/store intrinsics require that the last argument is a metadata string
; CHECK-NEXT: call void @llvm.amdgcn.global.store.b128({{.*}})
; CHECK-NEXT: metadata !0
define void @global_store_b128_01(ptr addrspace(1) %addr, <4 x i32> %data) {
entry:
  call void @llvm.amdgcn.global.store.b128(ptr addrspace(1) %addr, <4 x i32> %data, metadata !0)
  ret void
}

; ---------- invalid string metadata -------------------------------------------
; CHECK:      'wave' is not a valid scope for global load/store intrinsics
; CHECK-NEXT: call <4 x i32> @llvm.amdgcn.global.load.b128({{.*}})
; CHECK-NEXT: metadata !2
define <4 x i32> @global_load_b128_02(ptr addrspace(1) %addr) {
entry:
  %data = call <4 x i32> @llvm.amdgcn.global.load.b128(ptr addrspace(1) %addr, metadata !2)
  ret <4 x i32> %data
}

; CHECK:      'wave' is not a valid scope for global load/store intrinsics
; CHECK-NEXT: call void @llvm.amdgcn.global.store.b128({{.*}})
; CHECK-NEXT: metadata !2
define void @global_store_b128_02(ptr addrspace(1) %addr, <4 x i32> %data) {
entry:
  call void @llvm.amdgcn.global.store.b128(ptr addrspace(1) %addr, <4 x i32> %data, metadata !2)
  ret void
}


!0 = !{!1}
!1 = !{!""}

!2 = !{!"wave"}

!3 = !{i32 1}
