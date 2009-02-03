; RUN: llvm-as < %s | llc -enable-eh

target triple = "i686-pc-linux-gnu"

define fastcc void @bc__support__high_resolution_time__initialize_clock_rate() {
entry:
	invoke void asm "rdtsc\0A\09movl %eax, $0\0A\09movl %edx, $1", "=*imr,=*imr,~{dirflag},~{fpsr},~{flags},~{dx},~{ax}"( i32* null, i32* null )
			to label %.noexc unwind label %cleanup144

.noexc:		; preds = %entry
	ret void

cleanup144:		; preds = %entry
	unwind
}
