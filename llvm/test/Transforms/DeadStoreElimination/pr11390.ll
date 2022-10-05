; RUN: opt -basic-aa -dse -S < %s | FileCheck %s
; PR11390
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define fastcc void @cat_domain(ptr nocapture %name, ptr nocapture %domain, ptr 
nocapture %s) nounwind uwtable {
entry:
  %call = tail call i64 @strlen(ptr %name) nounwind readonly
  %call1 = tail call i64 @strlen(ptr %domain) nounwind readonly
  %add = add i64 %call, 1
  %add2 = add i64 %add, %call1
  %add3 = add i64 %add2, 1
  %call4 = tail call noalias ptr @malloc(i64 %add3) nounwind
  store ptr %call4, ptr %s, align 8
  %tobool = icmp eq ptr %call4, null
  br i1 %tobool, label %return, label %if.end

if.end:                                           ; preds = %entry
  tail call void @llvm.memcpy.p0.p0.i64(ptr %call4, ptr %name, i64 %call, i1 false)
  %arrayidx = getelementptr inbounds i8, ptr %call4, i64 %call
  store i8 46, ptr %arrayidx, align 1
; CHECK: store i8 46
  %add.ptr5 = getelementptr inbounds i8, ptr %call4, i64 %add
  tail call void @llvm.memcpy.p0.p0.i64(ptr %add.ptr5, ptr %domain, i64 %call1, i1 false)
  %arrayidx8 = getelementptr inbounds i8, ptr %call4, i64 %add2
  store i8 0, ptr %arrayidx8, align 1
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

declare i64 @strlen(ptr nocapture) nounwind readonly

declare noalias ptr @malloc(i64) nounwind

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
