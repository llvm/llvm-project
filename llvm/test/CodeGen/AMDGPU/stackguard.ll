; RUN: not llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; FIXME: To actually support stackguard, need to fix intrinsic to
; return pointer in any address space.

; CHECK: error: unable to lower stackguard
define i1 @test_stackguard(ptr %p1) {
  %p2 = call ptr @llvm.stackguard()
  %res = icmp ne ptr %p2, %p1
  ret i1 %res
}

declare ptr @llvm.stackguard()
