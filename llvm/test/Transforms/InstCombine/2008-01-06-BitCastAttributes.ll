; Ignore stderr, we expect warnings there
; RUN: opt < %s -instcombine 2> /dev/null -S | FileCheck %s

; CHECK-NOT: bitcast

define void @a() {
  ret void
}

define signext i32 @b(i32* inreg  %x)   {
  ret i32 0
}

define void @c(...) {
  ret void
}

define void @g(i32* %y) {
; CHECK-LABEL: @g(
; CHECK: call i64 bitcast (i32 (i32*)* @b to i64 (i32)*)(i32 0)
	%x = call i64 bitcast (i32 (i32*)* @b to i64 (i32)*)( i32 0 )		; <i64> [#uses=0]

; The rest should not have bitcasts remaining
; CHECK-NOT: bitcast
  call void bitcast (void ()* @a to void (i32*)*)( i32* noalias  %y )
  call <2 x i32> bitcast (i32 (i32*)* @b to <2 x i32> (i32*)*)( i32* inreg  null )		; <<2 x i32>>:1 [#uses=0]
  call void bitcast (void (...)* @c to void (i32)*)( i32 0 )
  call void bitcast (void (...)* @c to void (i32)*)( i32 zeroext  0 )
  ret void
}
