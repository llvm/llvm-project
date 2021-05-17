; RUN: llc < %s | FileCheck %s

declare i32 @__gxx_personality_v0(...) addrspace(0)
declare void @__cxa_call_unexpected(i8*)
declare void @llvm.donothing() readnone

; CHECK: f1
define void @f1() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK-NOT: donothing
  invoke void @llvm.donothing()
  to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
  unreachable
}

; CHECK: f2
define void @f2() nounwind {
entry:
; CHECK-NOT: donothing
  call void @llvm.donothing()
  ret void
}
