; This file is used by testlink1.ll, so it doesn't actually do anything itself
;
; RUN: echo
target datalayout = "e-p:32:32"
	%myint = type i16
	%struct1 = type { i32, ptr, ptr, ptr }
	%struct2 = type { %struct1 }

define internal void @f1(ptr %tty) {
loopentry.preheader:
	%tmp.2.i.i = getelementptr %struct1, ptr %tty, i64 0, i32 1		; <ptr> [#uses=1]
	%tmp.3.i.i = load volatile ptr, ptr %tmp.2.i.i		; <ptr> [#uses=0]
	ret void
}

