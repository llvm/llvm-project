; RUN: opt -passes="default<O2>" -disable-output %s
; REQUIRES: asserts
; PR42272

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = external global i32, align 4
@f = external global i32, align 4
@g = external global i32, align 4

define ptr @a(i32 %h) #0 {
entry:
  %h.addr = alloca i32, align 4
  store i32 %h, ptr %h.addr, align 4
  %tmp0 = load i32, ptr %h.addr, align 4
  switch i32 %tmp0, label %sw.default [
    i32 4, label %sw.bb
    i32 3, label %sw.bb1
    i32 2, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  %call = call i32 (...) @c()
  unreachable

sw.bb1:                                           ; preds = %entry
  %call2 = call i32 (...) @c()
  unreachable

sw.bb3:                                           ; preds = %entry
  %call4 = call i32 (...) @c()
  %conv = sext i32 %call4 to i64
  %tmp1 = inttoptr i64 %conv to ptr
  ret ptr %tmp1

sw.default:                                       ; preds = %entry
  unreachable
}

define i32 @m() #1 {
entry:
  %call = call i32 @j()
  %call1 = call i32 @j()
  ret i32 undef
}

define internal i32 @j() #0 {
entry:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %tmp0 = load i32, ptr %i, align 4
  %tmp1 = load i32, ptr @f, align 4
  %cmp = icmp ult i32 %tmp0, %tmp1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %tmp2 = load i32, ptr @f, align 4
  %call3 = call ptr @a(i32 %tmp2)
  ret i32 undef

for.body:                                         ; preds = %for.cond
  %call = call i32 (...) @c()
  %call1 = call i32 (...) @c()
  %call2 = call i32 (...) @c()
  %tmp3 = load i32, ptr @b, align 4
  %tmp4 = load i32, ptr @g, align 4
  %sub = sub nsw i32 %tmp4, %tmp3
  store i32 %sub, ptr @g, align 4
  %tmp5 = load i32, ptr %i, align 4
  %inc = add i32 %tmp5, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
}

declare i32 @c(...) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { "target-cpu"="x86-64" }
attributes #2 = { argmemonly nounwind }

