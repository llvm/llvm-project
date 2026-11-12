; Ignore stderr, we expect warnings there
; RUN: opt < %s -passes=instcombine 2> /dev/null -S | FileCheck %s

; CHECK-NOT: bitcast

define void @a() {
  ret void
}

define signext i32 @b(ptr inreg  %x)   {
  ret i32 0
}

define void @c(...) {
  ret void
}

define void @g(ptr %y) {
; CHECK-LABEL: @g(
; CHECK: call i64 @b(i32 0)
	%x = call i64 @b( i32 0 )		; <i64> [#uses=0]

; The rest should not have bitcasts remaining
; CHECK-NOT: bitcast
  call void @a( ptr noalias  %y )
  call <2 x i32> @b( ptr inreg  null )		; <<2 x i32>>:1 [#uses=0]
  call void @c( i32 0 )
  call void @c( i32 zeroext  0 )
  ret void
}
