; RUN: opt -S -aarch64-stack-tagging %s -o - | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android29"

@stackbuf = dso_local local_unnamed_addr global ptr null, align 8
@jbuf = dso_local global [32 x i64] zeroinitializer, align 8

declare void @may_jump()

define dso_local noundef i1 @_Z6targetv() sanitize_memtag {
entry:
  %buf = alloca [4096 x i8], align 1
  %call = call i32 @setjmp(ptr noundef @jbuf)
  switch i32 %call, label %while.body [
    i32 1, label %return
    i32 2, label %sw.bb1
  ]

sw.bb1:                                           ; preds = %entry
  br label %return

while.body:                                       ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 4096, ptr nonnull %buf) #10
  store ptr %buf, ptr @stackbuf, align 8
  ; may_jump may call longjmp, going back to the switch (and then the return),
  ; bypassing the lifetime.end. This is why we need to untag on the return,
  ; rather than the lifetime.end.
  call void @may_jump()
  call void @llvm.lifetime.end.p0(i64 4096, ptr nonnull %buf) #10
  br label %return

; CHECK-LABEL: return:
; CHECK: call void @llvm.aarch64.settag
return:                                           ; preds = %entry, %while.body, %sw.bb1
  %retval.0 = phi i1 [ true, %while.body ], [ true, %sw.bb1 ], [ false, %entry ]
  ret i1 %retval.0
}

declare i32 @setjmp(ptr noundef) returns_twice

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

