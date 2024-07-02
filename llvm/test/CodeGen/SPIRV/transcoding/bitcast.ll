; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Check the bitcast is translated back to bitcast

; CHECK: Bitcast

define spir_kernel void @test_fn(<2 x i8> addrspace(1)* nocapture readonly %src, i16 addrspace(1)* nocapture %dst) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %sext = shl i64 %call, 32
  %idxprom = ashr exact i64 %sext, 32
  %arrayidx = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %src, i64 %idxprom
  %0 = load <2 x i8>, <2 x i8> addrspace(1)* %arrayidx, align 2
  %astype = bitcast <2 x i8> %0 to i16
  %arrayidx2 = getelementptr inbounds i16, i16 addrspace(1)* %dst, i64 %idxprom
  store i16 %astype, i16 addrspace(1)* %arrayidx2, align 2
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
