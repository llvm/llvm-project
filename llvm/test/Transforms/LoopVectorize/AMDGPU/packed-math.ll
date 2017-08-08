; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s  -loop-vectorize -dce -instcombine -S | FileCheck -check-prefix=GFX9 -check-prefix=GCN %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=fiji < %s  -loop-vectorize -dce -instcombine -S | FileCheck -check-prefix=CIVI -check-prefix=GCN %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii < %s  -loop-vectorize -dce -instcombine -S | FileCheck -check-prefix=CIVI -check-prefix=GCN %s

; GCN-LABEL: @vectorize_v2f16_loop(
; GFX9: vector.body:
; GFX9: phi <2 x half>
; GFX9: load <2 x half>
; GFX9: fadd fast <2 x half>

; GFX9: middle.block:
; GFX9: fadd fast <2 x half>

; VI: phi half
; VI: phi load half
; VI: fadd fast half
define half @vectorize_v2f16_loop(half addrspace(1)* noalias %s) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %q.04 = phi half [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds half, half addrspace(1)* %s, i64 %indvars.iv
  %0 = load half, half addrspace(1)* %arrayidx, align 2
  %add = fadd fast half %q.04, %0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %add.lcssa = phi half [ %add, %for.body ]
  ret half %add.lcssa
}
