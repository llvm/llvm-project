; RUN: opt -passes='function-attrs,simplifycfg' -S %s | FileCheck %s

@vtable = global i8 0

declare i32 @__gxx_personality_v0(...)
declare void @target()

; CHECK: declare void @llvm.icall.branch.funnel(...){{$}}
declare void @llvm.icall.branch.funnel(...)

; CHECK-LABEL: define void @funnel(
; CHECK: musttail call void (...) @llvm.icall.branch.funnel
define void @funnel(ptr nest %vt, ...) {
entry:
  musttail call void (...) @llvm.icall.branch.funnel(
      ptr %vt, ptr @vtable, ptr @target, ...)
  ret void
}

; CHECK-LABEL: define{{.*}} i32 @caller()
; CHECK: invoke void (ptr, ...) @funnel(ptr nest @vtable)
; CHECK: exception:
define i32 @caller() personality ptr @__gxx_personality_v0 {
entry:
  invoke void (ptr, ...) @funnel(ptr nest @vtable)
      to label %normal unwind label %exception
normal:
  ret i32 0
exception:
  %lp = landingpad { ptr, i32 } cleanup
  ret i32 1
}
