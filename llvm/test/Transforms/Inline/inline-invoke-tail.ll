; RUN: opt < %s -inline -S | not grep "tail call void @llvm.memcpy.p0i8.p0i8.i32"
; PR3550

define internal void @foo(i32* %p, i32* %q) {
; CHECK-NOT: @foo
entry:
  %pp = bitcast i32* %p to i8*
  %qq = bitcast i32* %q to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %pp, i8* %qq, i32 4, i32 1, i1 false)
  ret void
}

define i32 @main() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: define i32 @main() personality i32 (...)* @__gxx_personality_v0
entry:
  %a = alloca i32
  %b = alloca i32
  store i32 1, i32* %a, align 4
  store i32 0, i32* %b, align 4
  invoke void @foo(i32* %a, i32* %b)
      to label %invcont unwind label %lpad
; CHECK-NOT: invoke
; CHECK-NOT: @foo
; CHECK-NOT: tail
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
; CHECK: br

invcont:
  %retval = load i32, i32* %a, align 4
  ret i32 %retval

lpad:
  %exn = landingpad {i8*, i32}
         catch i8* null
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
