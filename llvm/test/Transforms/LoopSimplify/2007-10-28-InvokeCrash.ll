; RUN: opt < %s -passes=loop-simplify -disable-output
; PR1752
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-s0:0:64-f80:32:32"
target triple = "i686-pc-mingw32"

define void @func() personality i32 (...)* @__gxx_personality_v0 {
bb_init:
	br label %bb_main

bb_main:
        br label %invcont17.normaldest

invcont17.normaldest917:		; No predecessors!
	%tmp23 = invoke i32 @foo()
			to label %invcont17.normaldest unwind label %invcont17.normaldest.normaldest

invcont17.normaldest:		; preds = %invcont17.normaldest917, %bb_main
	br label %bb_main

invcont17.normaldest.normaldest:		; No predecessors!
        %exn = landingpad {i8*, i32}
                 catch i8* null
        store i32 %tmp23, i32* undef
	br label %bb_main
}

declare i32 @foo()

declare i32 @__gxx_personality_v0(...)
