;; RUN: opt -S -passes=ripple %s | FileCheck %s

define dso_local void @masked_load_store(i32 noundef %N, ptr noundef readonly captures(none) %a, ptr noundef writeonly captures(none) %b) local_unnamed_addr {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i32(i32 0, i32 63, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1)
  %0 = tail call i32 @llvm.ripple.block.index.i32(ptr %BS, i32 0)
  %call = tail call i32 @foo(i32 noundef %0) #4
  %tobool.not = icmp eq i32 %call, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds nuw float, ptr %a, i32 %0
  %1 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds nuw float, ptr %b, i32 %0
  store float %1, ptr %arrayidx1, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
;; CHECK: void @masked_load_store
;; CHECK: call <63 x float> @llvm.masked.load.v63f32.p0
;; CHECK: call void @llvm.masked.store.v63f32.p0

declare dso_local i32 @foo(i32 noundef) local_unnamed_addr