; RUN: opt -passes=hwasan -hwasan-use-stack-safety=0 -hwasan-use-after-scope -S < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android29"

@stackbuf = dso_local local_unnamed_addr global i8* null, align 8
@jbuf = dso_local global [32 x i64] zeroinitializer, align 8

declare void @may_jump()

define dso_local noundef i1 @_Z6targetv() sanitize_hwaddress {
entry:
  %buf = alloca [4096 x i8], align 1
  %call = call i32 @setjmp(i64* noundef getelementptr inbounds ([32 x i64], [32 x i64]* @jbuf, i64 0, i64 0))
  switch i32 %call, label %while.body [
    i32 1, label %return
    i32 2, label %sw.bb1
  ]

sw.bb1:                                           ; preds = %entry
  br label %return

while.body:                                       ; preds = %entry
  %0 = getelementptr inbounds [4096 x i8], [4096 x i8]* %buf, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* nonnull %0) #10
  store i8* %0, i8** @stackbuf, align 8
  ; may_jump may call longjmp, going back to the switch (and then the return),
  ; bypassing the lifetime.end. This is why we need to untag on the return,
  ; rather than the lifetime.end.
  call void @may_jump()
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* nonnull %0) #10
  br label %return

; CHECK-LABEL: return:
; CHECK: void @llvm.memset.p0i8.i64({{.*}}, i8 0, i64 256, i1 false)
return:                                           ; preds = %entry, %while.body, %sw.bb1
  %retval.0 = phi i1 [ true, %while.body ], [ true, %sw.bb1 ], [ false, %entry ]
  ret i1 %retval.0
}

declare i32 @setjmp(i64* noundef) returns_twice

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
