; RUN: llvm-as -disable-output %s

define void @f() personality ptr @__gcc_personality_v0 {
  invoke void @llvm.bpf.unwind.exit()
          to label %cont unwind label %pad

cont:
  unreachable

pad:
  %lp = landingpad { ptr, i32 }
          cleanup
  ret void
}

declare i32 @__gcc_personality_v0(...)
