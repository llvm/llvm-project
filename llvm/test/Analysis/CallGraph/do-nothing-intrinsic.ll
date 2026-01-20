; RUN: opt < %s -passes='require<callgraph>'
; PR13903

define void @main() personality ptr @__gxx_personality_v0 {
  invoke void @llvm.donothing()
          to label %ret unwind label %unw
unw:
  %tmp = landingpad i8 cleanup
  br label %ret
ret:
  ret void
}
declare i32 @__gxx_personality_v0(...)
declare void @llvm.donothing() nounwind readnone
