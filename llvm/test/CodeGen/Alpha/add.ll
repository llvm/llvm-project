;test all the shifted and signextending adds and subs with and without consts
;
; RUN: llvm-as < %s | llc -march=alpha -o %t.s -f
; RUN: grep {	addl} %t.s | count 2
; RUN: grep {	addq} %t.s | count 2
; RUN: grep {	subl} %t.s | count 2
; RUN: grep {	subq} %t.s | count 1
;
; RUN: grep {lda \$0,-100(\$16)} %t.s | count 1
; RUN: grep {s4addl} %t.s | count 2
; RUN: grep {s8addl} %t.s | count 2
; RUN: grep {s4addq} %t.s | count 2
; RUN: grep {s8addq} %t.s | count 2
;
; RUN: grep {s4subl} %t.s | count 2
; RUN: grep {s8subl} %t.s | count 2
; RUN: grep {s4subq} %t.s | count 2
; RUN: grep {s8subq} %t.s | count 2


define i32 @al(i32 signext %x.s, i32 signext %y.s) signext {
entry:
	%tmp.3.s = add i32 %y.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @ali(i32 signext %x.s) signext {
entry:
	%tmp.3.s = add i32 100, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 @aq(i64 signext %x.s, i64 signext %y.s) signext {
entry:
	%tmp.3.s = add i64 %y.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 @aqi(i64 %x.s) {
entry:
	%tmp.3.s = add i64 100, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @sl(i32 signext %x.s, i32 signext %y.s) signext {
entry:
	%tmp.3.s = sub i32 %y.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @sli(i32 signext %x.s) signext {
entry:
	%tmp.3.s = sub i32 %x.s, 100		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 @sq(i64 %x.s, i64 %y.s) {
entry:
	%tmp.3.s = sub i64 %y.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 @sqi(i64 %x.s) {
entry:
	%tmp.3.s = sub i64 %x.s, 100		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @a4l(i32 signext %x.s, i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 2		; <i32> [#uses=1]
	%tmp.3.s = add i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @a8l(i32 signext %x.s, i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 3		; <i32> [#uses=1]
	%tmp.3.s = add i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 @a4q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 2		; <i64> [#uses=1]
	%tmp.3.s = add i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 @a8q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 3		; <i64> [#uses=1]
	%tmp.3.s = add i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @a4li(i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 2		; <i32> [#uses=1]
	%tmp.3.s = add i32 100, %tmp.1.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @a8li(i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 3		; <i32> [#uses=1]
	%tmp.3.s = add i32 100, %tmp.1.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 @a4qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 2		; <i64> [#uses=1]
	%tmp.3.s = add i64 100, %tmp.1.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 @a8qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 3		; <i64> [#uses=1]
	%tmp.3.s = add i64 100, %tmp.1.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @s4l(i32 signext %x.s, i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 2		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @s8l(i32 signext %x.s, i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 3		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 @s4q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 2		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 @s8q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 3		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @s4li(i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 2		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, 100		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @s8li(i32 signext %y.s) signext {
entry:
	%tmp.1.s = shl i32 %y.s, 3		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, 100		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 @s4qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 2		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, 100		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 @s8qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, 3		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, 100		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}
