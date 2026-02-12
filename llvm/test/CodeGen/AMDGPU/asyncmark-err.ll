; RUN: split-file %s %t
; RUN: not --crash llc -filetype=null -global-isel=0 -mtriple=amdgcn -mcpu=gfx1250 %t/mark.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not         llc -filetype=null -global-isel=1 -mtriple=amdgcn -mcpu=gfx1250 %t/mark.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not --crash llc -filetype=null -global-isel=0 -mtriple=amdgcn -mcpu=gfx1250 %t/wait.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not         llc -filetype=null -global-isel=1 -mtriple=amdgcn -mcpu=gfx1250 %t/wait.ll 2>&1 | FileCheck --ignore-case %s

; CHECK: LLVM ERROR: Cannot select

;--- mark.ll
define void @async_err() {
  call void @llvm.amdgcn.asyncmark()
  ret void
}

;--- wait.ll
define void @async_err() {
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  ret void
}
