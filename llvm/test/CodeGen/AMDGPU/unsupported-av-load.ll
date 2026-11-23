; RUN: not llc -global-isel=0 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

; RUN: not llc -global-isel=1 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

define void @av_load(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
; ERR: error: {{.*}}: in function @av_load {{.*}}: llvm.amdgcn.av.load.b8 requires target feature 'flat-global-insts'
; ERR: error: {{.*}}: in function @av_load {{.*}}: llvm.amdgcn.av.load.b16 requires target feature 'flat-global-insts'
; ERR: error: {{.*}}: in function @av_load {{.*}}: llvm.amdgcn.av.load.b32 requires target feature 'flat-global-insts'
; ERR: error: {{.*}}: in function @av_load {{.*}}: llvm.amdgcn.av.load.b64 requires target feature 'flat-global-insts'
; ERR: error: {{.*}}: in function @av_load {{.*}}: llvm.amdgcn.av.load.b128 requires target feature 'flat-global-insts'
entry:
  %data8 = call i8 @llvm.amdgcn.av.load.b8.p1(ptr addrspace(1) %addr, metadata !0)
  %data16 = call i16 @llvm.amdgcn.av.load.b16.p1(ptr addrspace(1) %addr, metadata !0)
  %data32 = call i32 @llvm.amdgcn.av.load.b32.p1(ptr addrspace(1) %addr, metadata !0)
  %data64 = call <2 x i32> @llvm.amdgcn.av.load.b64.p1(ptr addrspace(1) %addr, metadata !0)
  %data128 = call <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1) %addr, metadata !0)
  store volatile i8 %data8, ptr addrspace(1) %out
  store volatile i16 %data16, ptr addrspace(1) %out
  store volatile i32 %data32, ptr addrspace(1) %out
  store volatile <2 x i32> %data64, ptr addrspace(1) %out
  store volatile <4 x i32> %data128, ptr addrspace(1) %out
  ret void
}

!0 = !{!""}
