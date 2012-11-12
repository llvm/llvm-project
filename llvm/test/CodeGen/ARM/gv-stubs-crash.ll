; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=pic
; <rdar://problem/10336715>

@Exn = external hidden unnamed_addr constant { i8*, i8* }

define hidden void @func(i32* %this, i32* %e) optsize align 2 {
  %e.ld = load i32* %e, align 4
  %inv = invoke zeroext i1 @func2(i32* %this, i32 %e.ld) optsize
          to label %ret unwind label %lpad

ret:
  ret void

lpad:
  %lp = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
          catch i8* bitcast ({ i8*, i8* }* @Exn to i8*)
  br label %.loopexit4

.loopexit4:
  %exn = call i8* @__cxa_allocate_exception(i32 8) nounwind
  call void @__cxa_throw(i8* %exn, i8* bitcast ({ i8*, i8* }* @Exn to i8*), i8* bitcast (void (i32*)* @dtor to i8*)) noreturn
  unreachable

resume:
  resume { i8*, i32 } %lp
}

declare hidden zeroext i1 @func2(i32*, i32) optsize align 2

declare i8* @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_sj0(...)

declare void @dtor(i32*) optsize

declare void @__cxa_throw(i8*, i8*, i8*)
