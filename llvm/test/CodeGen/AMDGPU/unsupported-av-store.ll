; RUN: not llc -global-isel=0 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

; RUN: not llc -global-isel=1 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

define void @av_store(ptr addrspace(1) %addr, i16 %data16, i32 %data32, i64 %data64, <4 x i32> %data128) {
; ERR: error: {{.*}}: in function @av_store {{.*}}: llvm.amdgcn.av.store.b16 requires target feature 'flat-global-insts'
; ERR: error: {{.*}}: in function @av_store {{.*}}: llvm.amdgcn.av.store.b32 requires target feature 'flat-global-insts'
; ERR: error: {{.*}}: in function @av_store {{.*}}: llvm.amdgcn.av.store.b64 requires target feature 'flat-global-insts'
; ERR: error: {{.*}}: in function @av_store {{.*}}: llvm.amdgcn.av.store.b128 requires target feature 'flat-global-insts'
entry:
  call void @llvm.amdgcn.av.store.b16.p1(ptr addrspace(1) %addr, i16 %data16, metadata !0)
  call void @llvm.amdgcn.av.store.b32.p1(ptr addrspace(1) %addr, i32 %data32, metadata !0)
  call void @llvm.amdgcn.av.store.b64.p1(ptr addrspace(1) %addr, i64 %data64, metadata !0)
  call void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1) %addr, <4 x i32> %data128, metadata !0)
  ret void
}

!0 = !{!""}
