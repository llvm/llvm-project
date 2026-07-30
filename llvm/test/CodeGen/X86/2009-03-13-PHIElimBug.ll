; RUN: llc < %s -mtriple=i686-linux | FileCheck %s
; Check the register copy comes after the call to f and before the call to g
; PR3784

declare i32 @f()

declare i32 @g()

define i32 @phi() personality ptr @__gxx_personality_v0 {
entry:
	%a = call i32 @f()		; <i32> [#uses=1]
	%b = invoke i32 @g()
			to label %cont unwind label %lpad		; <i32> [#uses=1]

cont:		; preds = %entry
	%x = phi i32 [ %b, %entry ]		; <i32> [#uses=0]
	%aa = call i32 @g()		; <i32> [#uses=1]
	%bb = invoke i32 @g()
			to label %cont2 unwind label %lpad		; <i32> [#uses=1]

cont2:		; preds = %cont
	%xx = phi i32 [ %bb, %cont ]		; <i32> [#uses=1]
	ret i32 %xx

lpad:		; preds = %cont, %entry
	%y = phi i32 [ %a, %entry ], [ %aa, %cont ]		; <i32> [#uses=1]
        %exn = landingpad {ptr, i32}
                 cleanup
	ret i32 %y
}

; CHECK: call{{.*}}f
; CHECK: movl %eax, %esi
; CHECK: call{{.*}}g

declare i32 @__gxx_personality_v0(...)
