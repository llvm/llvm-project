; RUN: llc -no-integrated-as < %s

; XFAIL: avr

define fastcc void @bc__support__high_resolution_time__initialize_clock_rate() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void asm "rdtsc\0A\09movl %eax, $0\0A\09movl %edx, $1", "=*imr,=*imr,~{dirflag},~{fpsr},~{flags},~{dx},~{ax}"( i32* null, i32* null )
      to label %.noexc unwind label %cleanup144

.noexc:		; preds = %entry
  ret void

cleanup144:		; preds = %entry
  %exn = landingpad {i8*, i32}
            cleanup
  resume { i8*, i32 } %exn
}

declare i32 @__gxx_personality_v0(...)
