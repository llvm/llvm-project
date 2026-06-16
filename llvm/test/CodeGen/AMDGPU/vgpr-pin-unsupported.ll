; Pinning an i1 is rejected by the IR verifier: a lane mask cannot be made
; VGPR-resident, so the check lives in one place rather than scattered across
; the backend lowering.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not opt -passes=verify %s -disable-output 2>&1 | FileCheck %s

; CHECK: llvm.amdgcn.internal.vgpr.pin does not support i1 operands
; CHECK: llvm.amdgcn.internal.vgpr.pin does not support i1 operands

define amdgpu_kernel void @pin_i1(i1 %in) {
entry:
  call void @llvm.amdgcn.internal.vgpr.pin.i1(i1 %in)
  ret void
}

; The scalar-type check also rejects a vector of i1 (a lane mask).
define amdgpu_kernel void @pin_v4i1(<4 x i1> %in) {
entry:
  call void @llvm.amdgcn.internal.vgpr.pin.v4i1(<4 x i1> %in)
  ret void
}

declare void @llvm.amdgcn.internal.vgpr.pin.i1(i1)
declare void @llvm.amdgcn.internal.vgpr.pin.v4i1(<4 x i1>)
