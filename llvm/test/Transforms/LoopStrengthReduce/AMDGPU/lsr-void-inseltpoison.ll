; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

@array = external addrspace(4) constant [32 x [800 x i32]], align 4

; GCN-LABEL: {{^}}test_lsr_voidty:
define amdgpu_kernel void @test_lsr_voidty() {
entry:
  br label %for.body

for.body:                                 ; preds = %for.body.i, %entry
  br label %for.body.i

for.body.i:                               ; preds = %for.body.i, %for.body
  %ij = phi i32 [ 0, %for.body ], [ %inc14, %for.body.i ]
  %tmp = load i32, i32 addrspace(5)* undef, align 4
  %inc13 = or i32 %ij, 2
  %shl = shl i32 1, 0
  %and = and i32 %shl, %tmp
  %tobool = icmp eq i32 %and, 0
  %add = mul nuw nsw i32 %inc13, 5
  %tmp1 = zext i32 %add to i64
  %arrayidx8 = getelementptr inbounds [32 x [800 x i32]], [32 x [800 x i32]] addrspace(4)* @array, i64 0, i64 undef, i64 %tmp1
  %tmp2 = load i32, i32 addrspace(4)* %arrayidx8, align 4
  %and9 = select i1 %tobool, i32 0, i32 %tmp2
  %xor = xor i32 undef, %and9
  %inc1 = or i32 %ij, 3
  %add2 = mul nuw nsw i32 %inc1, 5
  %add6 = add nuw nsw i32 %add2, 1
  %tmp3 = zext i32 %add6 to i64
  %arrayidx9 = getelementptr inbounds [32 x [800 x i32]], [32 x [800 x i32]] addrspace(4)* @array, i64 0, i64 undef, i64 %tmp3
  %tmp4 = bitcast i32 addrspace(4)* %arrayidx9 to <4 x i32> addrspace(4)*
  %tmp5 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp4, align 4
  %reorder_shuffle2 = shufflevector <4 x i32> %tmp5, <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %tmp6 = select <4 x i1> undef, <4 x i32> zeroinitializer, <4 x i32> %reorder_shuffle2
  %inc14 = add nuw nsw i32 %ij, 4
  br i1 undef, label %for.body, label %for.body.i
}
