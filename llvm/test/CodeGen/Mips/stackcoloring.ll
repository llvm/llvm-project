; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s

@g1 = external global i32*

; CHECK-LABEL: foo1:
; CHECK: lw ${{[0-9]+}}, %got(g1)
; CHECK: # %for.body
; CHECK: # %for.end

define i32 @foo1() {
entry:
  %b = alloca [16 x i32], align 4
  %0 = bitcast [16 x i32]* %b to i8*
  call void @llvm.lifetime.start(i64 64, i8* %0)
  %arraydecay = getelementptr inbounds [16 x i32], [16 x i32]* %b, i32 0, i32 0
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %v.04 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %1 = load i32*, i32** @g1, align 4
  %arrayidx = getelementptr inbounds i32, i32* %1, i32 %i.05
  %2 = load i32, i32* %arrayidx, align 4
  %call = call i32 @foo2(i32 %2, i32* %arraydecay)
  %add = add nsw i32 %call, %v.04
  %inc = add nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  call void @llvm.lifetime.end(i64 64, i8* %0)
  ret i32 %add
}

declare void @llvm.lifetime.start(i64, i8* nocapture)

declare i32 @foo2(i32, i32*)

declare void @llvm.lifetime.end(i64, i8* nocapture)
