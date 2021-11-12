; RUN: opt < %s -O3 -S > %t
; RUN:   grep 'constant i32 undef' %t | count 1
; RUN:   grep 'constant i32 5' %t | count 1
; RUN:   grep 'i32 7' %t | count 1
; RUN:   grep 'i32 9' %t | count 1

	%0 = type { i32, i32 }		; type %0
@a = weak constant i32 undef		; <i32*> [#uses=1]
@b = weak constant i32 5		; <i32*> [#uses=1]
@c = weak constant %0 { i32 7, i32 9 }		; <%0*> [#uses=1]

define i32 @la() {
	%v = load i32, i32* @a		; <i32> [#uses=1]
	ret i32 %v
}

define i32 @lb() {
	%v = load i32, i32* @b		; <i32> [#uses=1]
	ret i32 %v
}

define i32 @lc() {
	%g = getelementptr %0, %0* @c, i32 0, i32 0		; <i32*> [#uses=1]
	%u = load i32, i32* %g		; <i32> [#uses=1]
	%h = getelementptr %0, %0* @c, i32 0, i32 1		; <i32*> [#uses=1]
	%v = load i32, i32* %h		; <i32> [#uses=1]
	%r = add i32 %u, %v
	ret i32 %r
}

define i32 @f() {
	%u = call i32 @la()		; <i32> [#uses=1]
	%v = call i32 @lb()		; <i32> [#uses=1]
	%w = call i32 @lc()		; <i32> [#uses=1]
	%r = add i32 %u, %v		; <i32> [#uses=1]
	%s = add i32 %r, %w		; <i32> [#uses=1]
	ret i32 %s
}
