; RUN: opt < %s -deadargelim -dce -S > %t
; RUN: cat %t | grep 123

; This test tries to catch wrongful removal of return values for a specific case
; that was breaking llvm-gcc builds.

; This function has a live return value, it is used by @alive.
define internal i32 @test5() {
  ret i32 123 
}

; This function doesn't use the return value @test5 and tries to lure DAE into
; marking @test5's return value dead because only this call is unused.
define i32 @dead() {
  %DEAD = call i32 @test5()
  ret i32 0
}

; This function ensures the retval of @test5 is live.
define i32 @alive() {
  %LIVE = call i32 @test5()
  ret i32 %LIVE
}
