; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42 -filetype=null < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu9.42 -filetype=null < %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}}: in function av_load_bad_scope {{.*}}: Unsupported atomic synchronization scope
; CHECK: error: {{.*}}: in function av_store_bad_scope {{.*}}: Unsupported atomic synchronization scope

define <4 x i32> @av_load_bad_scope(ptr addrspace(1) %p) {
  %v = call <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1) %p, metadata !0)
  ret <4 x i32> %v
}

define void @av_store_bad_scope(ptr addrspace(1) %p, <4 x i32> %v) {
  call void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1) %p, <4 x i32> %v, metadata !0)
  ret void
}

!0 = !{!"xyzzy"}
