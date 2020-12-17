; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -loop-vectorize -simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -check-prefixes=GCN %s
; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -loop-vectorize -pass-remarks-analysis='loop-vectorize' < %s 2>&1 | FileCheck -check-prefixes=REMARK %s

; GCN-LABEL: @runtime_check_divergent_target(
; GCN-NOT: load <2 x half>
; GCN-NOT: store <2 x half>

; REMARK: remark: <unknown>:0:0: loop not vectorized: runtime pointer checks needed. Not enabled for divergent target
define amdgpu_kernel void @runtime_check_divergent_target(half addrspace(1)* nocapture %a, half addrspace(1)* nocapture %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds half, half addrspace(1)* %b, i64 %indvars.iv
  %load = load half, half addrspace(1)* %arrayidx, align 4
  %mul = fmul half %load, 3.0
  %arrayidx2 = getelementptr inbounds half, half addrspace(1)* %a, i64 %indvars.iv
  store half %mul, half addrspace(1)* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind }
