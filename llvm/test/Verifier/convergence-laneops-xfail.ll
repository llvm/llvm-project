; RUN: not --crash llc -stop-after=amdgpu-isel -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs -o - %s 2>&1 | FileCheck %s

; CHECK: *** Bad machine code: Cannot mix controlled and uncontrolled convergence in the same function. ***
; CHECK: function:    basic_branch_i64
define i64 @basic_branch_i64(i64 %src, i1 %cond) #0 {
entry:
  %t = call token @llvm.experimental.convergence.anchor()
  %x = add i64 %src, 1
  br i1 %cond, label %then, label %else

then:
  %r = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %x) [ "convergencectrl"(token %t) ]
  br label %else

else:
  %p = phi i64 [%r, %then], [%x, %entry]
  ret i64 %p
}

