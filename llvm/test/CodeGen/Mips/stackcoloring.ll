; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s

@g1 = external global ptr

; CHECK-LABEL: foo1:
; CHECK: lw ${{[0-9]+}}, %got(g1)
; CHECK: # %for.body
; CHECK: # %for.end

define i32 @foo1() {
entry:
  %b = alloca [16 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 64, ptr %b)
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %v.04 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %0 = load ptr, ptr @g1, align 4
  %arrayidx = getelementptr inbounds i32, ptr %0, i32 %i.05
  %1 = load i32, ptr %arrayidx, align 4
  %call = call i32 @foo2(i32 %1, ptr %b)
  %add = add nsw i32 %call, %v.04
  %inc = add nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  call void @llvm.lifetime.end.p0(i64 64, ptr %b)
  ret i32 %add
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare i32 @foo2(i32, ptr)

declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
