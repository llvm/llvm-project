; RUN: llc -march=amdgcn -mattr=-promote-alloca -verify-machineinstrs < %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -mattr=-promote-alloca -verify-machineinstrs < %s

; Test that CopyToReg instructions don't have non-register operands prior
; to being emitted.

; Make sure this doesn't crash
; CHECK-LABEL: {{^}}copy_to_reg_frameindex:
define amdgpu_kernel void @copy_to_reg_frameindex(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c) {
entry:
  %alloca = alloca [16 x i32], addrspace(5)
  br label %loop

loop:
  %inc = phi i32 [0, %entry], [%inc.i, %loop]
  %ptr = getelementptr [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %inc
  store i32 %inc, ptr addrspace(5) %ptr
  %inc.i = add i32 %inc, 1
  %cnd = icmp uge i32 %inc.i, 16
  br i1 %cnd, label %done, label %loop

done:
  %tmp1 = load i32, ptr addrspace(5) %alloca
  store i32 %tmp1, ptr addrspace(1) %out
  ret void
}
