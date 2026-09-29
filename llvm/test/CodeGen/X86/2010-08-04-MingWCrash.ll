; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=i386-pc-mingw32

define void @func() nounwind personality ptr @__gxx_personality_v0 {
invoke.cont:
  %call = tail call ptr @malloc()
  %a = invoke i32 @bar()
          to label %bb1 unwind label %lpad

bb1:
  ret void

lpad:
  %exn.ptr = landingpad { ptr, i32 }
           catch ptr null
  %exn = extractvalue { ptr, i32 } %exn.ptr, 0
  %eh.selector = extractvalue { ptr, i32 } %exn.ptr, 1
  %ehspec.fails = icmp slt i32 %eh.selector, 0
  br i1 %ehspec.fails, label %ehspec.unexpected, label %cleanup

cleanup:
  resume { ptr, i32 } %exn.ptr

ehspec.unexpected:
  tail call void @__cxa_call_unexpected(ptr %exn) noreturn nounwind
  unreachable
}

declare noalias ptr @malloc()

declare i32 @__gxx_personality_v0(...)

declare void @_Unwind_Resume_or_Rethrow(ptr)

declare void @__cxa_call_unexpected(ptr)

declare i32 @bar()
