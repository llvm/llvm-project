; RUN: opt < %s -inline -S | not grep "invoke void asm"
; PR1335

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
	%struct.gnat__strings__string_access = type { i8*, %struct.string___XUB* }
	%struct.string___XUB = type { i32, i32 }

define void @bc__support__high_resolution_time__clock() {
entry:
	call void asm "rdtsc\0A\09movl %eax, $0\0A\09movl %edx, $1", "=*imr,=*imr,~{dirflag},~{fpsr},~{flags},~{dx},~{ax}"( i32* null, i32* null ) nounwind
	unreachable
}

define fastcc void @bc__support__high_resolution_time__initialize_clock_rate() {
entry:
	invoke void @gnat__os_lib__getenv( %struct.gnat__strings__string_access* null )
			to label %invcont unwind label %cleanup144

invcont:		; preds = %entry
	invoke void @ada__calendar__delays__delay_for( )
			to label %invcont64 unwind label %cleanup144

invcont64:		; preds = %invcont
	invoke void @ada__calendar__clock( )
			to label %invcont65 unwind label %cleanup144

invcont65:		; preds = %invcont64
	invoke void @bc__support__high_resolution_time__clock( )
			to label %invcont67 unwind label %cleanup144

invcont67:		; preds = %invcont65
	ret void

cleanup144:		; preds = %invcont65, %invcont64, %invcont, %entry
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
	resume { i8*, i32 } %exn
}

declare i32 @__gxx_personality_v0(...)

declare void @gnat__os_lib__getenv(%struct.gnat__strings__string_access*)

declare void @ada__calendar__delays__delay_for()

declare void @ada__calendar__clock()

define void @bc__support__high_resolution_time___elabb() {
entry:
	call fastcc void @bc__support__high_resolution_time__initialize_clock_rate( )
	ret void
}
