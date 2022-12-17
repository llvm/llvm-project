; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s -passes=loop-vectorize -S | FileCheck -check-prefix=GFX90A %s

; GFX90A-LABEL: @vectorize_v2f32_loop(
; GFX90A-COUNT-2: load <2 x float>
; GFX90A-COUNT-2: fadd fast <2 x float>

define float @vectorize_v2f32_loop(float addrspace(1)* noalias %s) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %q.04 = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %s, i64 %indvars.iv
  %load = load float, float addrspace(1)* %arrayidx, align 4
  %add = fadd fast float %q.04, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %add.lcssa = phi float [ %add, %for.body ]
  ret float %add.lcssa
}
