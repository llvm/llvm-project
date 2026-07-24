; RUN: opt -mtriple=amdgpu12.51-amd-amdhsa < %s -passes=loop-vectorize -S | FileCheck -check-prefix=GFX1251 %s

; GFX1251-LABEL: @vectorize_v2f64_loop(
; GFX1251-COUNT-2: load <2 x double>
; GFX1251-COUNT-2: fadd fast <2 x double>

define double @vectorize_v2f64_loop(ptr addrspace(1) noalias %s) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %q.04 = phi double [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr addrspace(1) %s, i64 %indvars.iv
  %load = load double, ptr addrspace(1) %arrayidx, align 8
  %add = fadd fast double %q.04, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %add.lcssa = phi double [ %add, %for.body ]
  ret double %add.lcssa
}
