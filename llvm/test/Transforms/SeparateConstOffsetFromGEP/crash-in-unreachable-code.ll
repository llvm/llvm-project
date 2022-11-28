; RUN: opt -mtriple=amdgcn-amd-amdhsa -separate-const-offset-from-gep %s

@gv = external local_unnamed_addr addrspace(3) global [16 x i8], align 16

; The add referencing itself is illegal, except it's in an unreachable block.
define weak amdgpu_kernel void @foo() {
entry:
  ret void

for.body28.i:                                     ; preds = %for.body28.i
  %arrayidx3389.i = getelementptr inbounds [16 x i8], ptr addrspace(3) @gv, i32 0, i32 %inc38.7.i.1
  %inc38.7.i.1 = add nuw nsw i32 %inc38.7.i.1, 16
  br label %for.body28.i
}
