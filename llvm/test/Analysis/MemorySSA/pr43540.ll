; RUN: opt -S -passes=licm %s | FileCheck %s
@v_1 = global i8 0, align 1
@v_2 =  global i8 0, align 1

; CHECK-LABEL: @foo()
; CHECK: for.cond:
; CHECK-NOT: store
; CHECK: for.body:
; CHECK: call void @llvm.memcpy.p0.p0.i64
; CHECK: store
define void @foo() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %0 = phi i16 [ %inc, %for.body ], [ 0, %entry ]
  %cmp = icmp slt i16 %0, 1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @llvm.memcpy.p0.p0.i64(ptr @v_1, ptr @v_2, i64 1, i1 false)
  store i8 1, ptr @v_2, align 1
  %inc = add nsw i16 %0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

attributes #2 = { argmemonly nounwind willreturn }

