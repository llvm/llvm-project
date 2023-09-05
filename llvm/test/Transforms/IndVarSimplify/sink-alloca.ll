; RUN: opt < %s -passes=indvars -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

; PR4775
; Indvars shouldn't sink the alloca out of the entry block, even though
; it's not used until after the loop.
define i32 @main() nounwind {
; CHECK: entry:
; CHECK-NEXT: %result.i = alloca i32, align 4
entry:
  %result.i = alloca i32, align 4                 ; <ptr> [#uses=2]
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %call = call i32 @bar() nounwind                ; <i32> [#uses=1]
  %tobool = icmp eq i32 %call, 0                  ; <i1> [#uses=1]
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  store volatile i32 0, ptr %result.i
  %tmp.i = load volatile i32, ptr %result.i           ; <i32> [#uses=0]
  ret i32 0
}
declare i32 @bar()

; <rdar://problem/10352360>
; Indvars shouldn't sink the first alloca between the stacksave and stackrestore
; intrinsics.
declare ptr @a(...)
declare ptr @llvm.stacksave() nounwind
declare void @llvm.stackrestore(ptr) nounwind
define void @h(i64 %n) nounwind uwtable ssp {
; CHECK: entry:
; CHECK-NEXT: %vla = alloca ptr
; CHECK-NEXT: %savedstack = call ptr @llvm.stacksave.p0()
entry:
  %vla = alloca ptr, i64 %n, align 16
  %savedstack = call ptr @llvm.stacksave() nounwind
  %vla.i = alloca ptr, i64 %n, align 16
  br label %for.body.i

for.body.i:
  %indvars.iv37.i = phi i64 [ %indvars.iv.next38.i, %for.body.i ], [ 0, %entry ]
  %call.i = call ptr (...) @a() nounwind
  %arrayidx.i = getelementptr inbounds ptr, ptr %vla.i, i64 %indvars.iv37.i
  store ptr %call.i, ptr %arrayidx.i, align 8
  %indvars.iv.next38.i = add i64 %indvars.iv37.i, 1
  %exitcond5 = icmp eq i64 %indvars.iv.next38.i, %n
  br i1 %exitcond5, label %g.exit, label %for.body.i

g.exit:
  call void @llvm.stackrestore(ptr %savedstack) nounwind
  %call1 = call ptr (...) @a(ptr %vla) nounwind
  ret void
}
