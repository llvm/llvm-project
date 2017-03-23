; RUN: llc -march=amdgcn -mcpu=SI < %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s
; CHECK: s_endpgm

@gv = external unnamed_addr addrspace(2) constant [239 x i32], align 4

define amdgpu_kernel void @opencv_cvtfloat_crash(i32 addrspace(1)* %out, i32 %x) nounwind {
  %val = load i32, i32 addrspace(2)* getelementptr ([239 x i32], [239 x i32] addrspace(2)* @gv, i64 0, i64 239), align 4
  %mul12 = mul nsw i32 %val, 7
  br i1 undef, label %exit, label %bb

bb:
  %cmp = icmp slt i32 %x, 0
  br label %exit

exit:
  ret void
}

