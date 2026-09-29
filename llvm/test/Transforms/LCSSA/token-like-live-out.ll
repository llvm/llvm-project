; RUN: opt -passes='lcssa,verify' -verify-loop-info -S < %s | FileCheck %s

; Token-like target extension values cannot be used in PHI nodes. Make sure
; LCSSA formation leaves a live-out token-like value unchanged.

define void @target_type_live_out() {
; CHECK-LABEL: define void @target_type_live_out(
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %token = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") poison)
  %idx.next = add i32 %idx, 1
  %continue = icmp slt i32 %idx.next, 100
  br i1 %continue, label %loop, label %exit

exit:
; CHECK: exit:
; CHECK-NEXT: call target("amdgpu.stridemark") @llvm.ssa.copy.tamdgpu.stridemarkt(target("amdgpu.stridemark") %token)
; CHECK-NEXT: ret void
  %use = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") %token)
  ret void
}
