; RUN: opt -basicaa -dse -S < %s | FileCheck %s
; PR11390
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define fastcc void @cat_domain(i8* nocapture %name, i8* nocapture %domain, i8** 
nocapture %s) nounwind uwtable {
entry:
  %call = tail call i64 @strlen(i8* %name) nounwind readonly
  %call1 = tail call i64 @strlen(i8* %domain) nounwind readonly
  %add = add i64 %call, 1
  %add2 = add i64 %add, %call1
  %add3 = add i64 %add2, 1
  %call4 = tail call noalias i8* @malloc(i64 %add3) nounwind
  store i8* %call4, i8** %s, align 8
  %tobool = icmp eq i8* %call4, null
  br i1 %tobool, label %return, label %if.end

if.end:                                           ; preds = %entry
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %call4, i8* %name, i64 %call, i1 false)
  %arrayidx = getelementptr inbounds i8, i8* %call4, i64 %call
  store i8 46, i8* %arrayidx, align 1
; CHECK: store i8 46
  %add.ptr5 = getelementptr inbounds i8, i8* %call4, i64 %add
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %add.ptr5, i8* %domain, i64 %call1, i1 false)
  %arrayidx8 = getelementptr inbounds i8, i8* %call4, i64 %add2
  store i8 0, i8* %arrayidx8, align 1
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

declare i64 @strlen(i8* nocapture) nounwind readonly

declare noalias i8* @malloc(i64) nounwind

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
