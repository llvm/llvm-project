; RUN: opt < %s -passes=inline -S | not grep "tail call void @llvm.memcpy.p0.p0.i32"
; PR3550

define internal void @foo(ptr %p, ptr %q) {
; CHECK-NOT: @foo
entry:
  tail call void @llvm.memcpy.p0.p0.i32(ptr %p, ptr %q, i32 4, i1 false)
  ret void
}

define i32 @main() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define i32 @main() personality ptr @__gxx_personality_v0
entry:
  %a = alloca i32
  %b = alloca i32
  store i32 1, ptr %a, align 4
  store i32 0, ptr %b, align 4
  invoke void @foo(ptr %a, ptr %b)
      to label %invcont unwind label %lpad
; CHECK-NOT: invoke
; CHECK-NOT: @foo
; CHECK-NOT: tail
; CHECK: call void @llvm.memcpy.p0.p0.i32
; CHECK: br

invcont:
  %retval = load i32, ptr %a, align 4
  ret i32 %retval

lpad:
  %exn = landingpad {ptr, i32}
         catch ptr null
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
