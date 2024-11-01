; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; CHECK-LABEL: define void @foo
define void @foo() personality ptr @__gxx_personality_v0 {
entry:
; CHECK: alloca i8
; CHECK: call void @llvm.lifetime.start.p0
; CHECK: call void @bar()
; CHECK: call void @llvm.lifetime.end.p0
; CHECK: ret void
  %a = alloca i8
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %a) nounwind
  invoke void @bar() to label %invoke.cont unwind label %lpad

invoke.cont:
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %a) nounwind
  ret void

lpad:
; CHECK-NOT: landingpad
  %b = landingpad { ptr, i32 }
          cleanup
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %a) nounwind
  resume { ptr, i32 } %b
}

declare void @bar()

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) nounwind

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) nounwind

declare i32 @__gxx_personality_v0(...)
