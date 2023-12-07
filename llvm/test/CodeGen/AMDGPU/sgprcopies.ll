; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}checkTwoBlocksWithUniformBranch
; GCN: BB0_2
; GCN: v_add
define amdgpu_kernel void @checkTwoBlocksWithUniformBranch(ptr addrspace(1) nocapture %out, i32 %width, float %xPos, float %yPos, float %xStep, float %yStep, i32 %maxIter) {
entry:
  %conv = call i32 @llvm.amdgcn.workitem.id.x() #1
  %rem = urem i32 %conv, %width
  %div = udiv i32 %conv, %width
  %conv1 = sitofp i32 %rem to float
  %x = tail call float @llvm.fmuladd.f32(float %xStep, float %conv1, float %xPos)
  %conv2 = sitofp i32 %div to float
  %y = tail call float @llvm.fmuladd.f32(float %yStep, float %conv2, float %yPos)
  %yy = fmul float %y, %y
  %xy = tail call float @llvm.fmuladd.f32(float %x, float %x, float %yy)
  %cmp01 = fcmp ole float %xy, 4.000000e+00
  %cmp02 = icmp ne i32 %maxIter, 0
  %cond01 = and i1 %cmp02, %cmp01
  br i1 %cond01, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %x_val = phi float [ %call8, %for.body ], [ %x, %for.body.preheader ]
  %iter_val = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %y_val = phi float [ %call9, %for.body ], [ %y, %for.body.preheader ]
  %sub = fsub float -0.000000e+00, %y_val
  %call7 = tail call float @llvm.fmuladd.f32(float %x_val, float %x_val, float %x) #1
  %call8 = tail call float @llvm.fmuladd.f32(float %sub, float %y_val, float %call7) #1
  %mul = fmul float %x_val, 2.000000e+00
  %call9 = tail call float @llvm.fmuladd.f32(float %mul, float %y_val, float %y) #1
  %inc = add nuw i32 %iter_val, 1
  %mul3 = fmul float %call9, %call9
  %0 = tail call float @llvm.fmuladd.f32(float %call8, float %call8, float %mul3)
  %cmp = fcmp ole float %0, 4.000000e+00
  %cmp5 = icmp ult i32 %inc, %maxIter
  %or.cond = and i1 %cmp5, %cmp
  br i1 %or.cond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %iter.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %for.end.loopexit ]
  %idxprom = ashr exact i32 %conv, 32
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %idxprom
  store i32 %iter.0.lcssa, ptr addrspace(1) %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #0
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { nounwind readnone }
attributes #1 = { readnone }
