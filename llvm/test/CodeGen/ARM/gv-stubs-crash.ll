; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=pic
; <rdar://problem/10336715>

@Exn = external hidden unnamed_addr constant { ptr, ptr }

define hidden void @func(ptr %this, ptr %e) optsize align 2 personality ptr @__gxx_personality_sj0 {
  %e.ld = load i32, ptr %e, align 4
  %inv = invoke zeroext i1 @func2(ptr %this, i32 %e.ld) optsize
          to label %ret unwind label %lpad

ret:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          catch ptr @Exn
  br label %.loopexit4

.loopexit4:
  %exn = call ptr @__cxa_allocate_exception(i32 8) nounwind
  call void @__cxa_throw(ptr %exn, ptr @Exn, ptr @dtor) noreturn
  unreachable

resume:
  resume { ptr, i32 } %lp
}

declare hidden zeroext i1 @func2(ptr, i32) optsize align 2

declare ptr @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_sj0(...)

declare void @dtor(ptr) optsize

declare void @__cxa_throw(ptr, ptr, ptr)
