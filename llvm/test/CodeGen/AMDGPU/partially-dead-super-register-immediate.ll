; RUN: llc -march=amdgcn -verify-machineinstrs -verify-coalescing < %s

; The original and requires materializing a 64-bit immediate for
; s_and_b64. This is split into 2 x v_and_i32, part of the immediate
; is folded through the reg_sequence into the v_and_i32 operand, and
; only half of the result is ever used.
;
; During live interval construction, the first sub register def is
; incorrectly marked as dead.

declare i32 @llvm.amdgcn.workitem.id.x() #1

define amdgpu_kernel void @dead_def_subregister(i32 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep

  %lshr = shl i64 %val, 24
  %and1 = and i64 %lshr, 2190433320969 ; (255 << 33) | 9
  %vec = bitcast i64 %and1 to <2 x i32>
  %elt1 = extractelement <2 x i32> %vec, i32 1

  store i32 %elt1, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
