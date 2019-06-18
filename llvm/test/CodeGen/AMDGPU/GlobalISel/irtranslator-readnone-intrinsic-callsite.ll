; RUN: llc -mtriple=amdgcn-amd-amdhsa -global-isel -stop-after=irtranslator -o - %s | FileCheck %s

; Make sure that an intrinsic declaration that has side effects, but
; called with a readnone call site is translated to
; G_INTRINSIC_W_SIDE_EFFECTS

; CHECK-LABEL: name: getreg_callsite_attributes
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.s.getreg)
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.s.getreg)
define amdgpu_kernel void @getreg_callsite_attributes() {
  %reg0 = call i32 @llvm.amdgcn.s.getreg(i32 0)
  store volatile i32 %reg0, i32 addrspace(1)* undef
  %reg1 = call i32 @llvm.amdgcn.s.getreg(i32 0) #1
  store volatile i32 %reg1, i32 addrspace(1)* undef
  ret void
}

declare i32 @llvm.amdgcn.s.getreg(i32) #0

attributes #0 = { nounwind readonly inaccessiblememonly }
attributes #1 = { nounwind readnone }
