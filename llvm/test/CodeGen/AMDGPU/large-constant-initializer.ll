; RUN: llc -march=amdgcn -mcpu=tahiti < %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s
; CHECK: s_endpgm

@gv = external unnamed_addr addrspace(4) constant [239 x i32], align 4

define amdgpu_kernel void @opencv_cvtfloat_crash(ptr addrspace(1) %out, i32 %x, i1 %c0) nounwind {
  %val = load i32, ptr addrspace(4) getelementptr ([239 x i32], ptr addrspace(4) @gv, i64 0, i64 239), align 4
  %mul12 = mul nsw i32 %val, 7
  br i1 %c0, label %exit, label %bb

bb:
  %cmp = icmp slt i32 %x, 0
  br label %exit

exit:
  ret void
}

