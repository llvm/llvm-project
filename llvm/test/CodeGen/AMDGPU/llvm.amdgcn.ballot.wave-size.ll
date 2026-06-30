; RUN: split-file %s %t
; RUN: not llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %t/ballot-i32.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -global-isel=1 -global-isel-abort=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %t/ballot-i32.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %t/ballot-i64.ll
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %t/ballot-i64.ll
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -filetype=null %t/ballot-i32.ll
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -filetype=null %t/ballot-i32.ll

; ERR: error: {{.*}}: in function {{@?}}ballot_i32{{.*}}: llvm.amdgcn.ballot requires target feature 'wavefrontsize32'

;--- ballot-i32.ll
declare i32 @llvm.amdgcn.ballot.i32(i1)

define amdgpu_kernel void @ballot_i32(i32 %x, ptr addrspace(1) %out) {
  %trunc = trunc i32 %x to i1
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 %trunc)
  store i32 %ballot, ptr addrspace(1) %out
  ret void
}

;--- ballot-i64.ll
declare i64 @llvm.amdgcn.ballot.i64(i1)

define amdgpu_kernel void @ballot_i64(i32 %x, ptr addrspace(1) %out) {
  %trunc = trunc i32 %x to i1
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %trunc)
  store i64 %ballot, ptr addrspace(1) %out
  ret void
}
