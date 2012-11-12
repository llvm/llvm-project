; Test that FFSOpt works correctly
; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

; CHECK-NOT: call{{.*}}@ffs

@non_const = external global i32		; <i32*> [#uses=1]

declare i32 @ffs(i32)

declare i32 @ffsl(i32)

declare i32 @ffsll(i64)

define i32 @main() {
	%arg = load i32* @non_const		; <i32> [#uses=1]
	%val0 = call i32 @ffs( i32 %arg )		; <i32> [#uses=1]
	%val1 = call i32 @ffs( i32 1 )		; <i32> [#uses=1]
	%val2 = call i32 @ffs( i32 2048 )		; <i32> [#uses=1]
	%val3 = call i32 @ffsl( i32 65536 )		; <i32> [#uses=1]
	%val4 = call i32 @ffsll( i64 1024 )		; <i32> [#uses=1]
	%val5 = call i32 @ffsll( i64 17179869184 )		; <i32> [#uses=1]
	%val6 = call i32 @ffsll( i64 1152921504606846976 )		; <i32> [#uses=1]
	%rslt1 = add i32 %val1, %val2		; <i32> [#uses=1]
	%rslt2 = add i32 %val3, %val4		; <i32> [#uses=1]
	%rslt3 = add i32 %val5, %val6		; <i32> [#uses=1]
	%rslt4 = add i32 %rslt1, %rslt2		; <i32> [#uses=1]
	%rslt5 = add i32 %rslt4, %rslt3		; <i32> [#uses=2]
	%rslt6 = add i32 %rslt5, %val0		; <i32> [#uses=0]
	ret i32 %rslt5
}


; PR4206
define i32 @a(i64) nounwind {
        %2 = call i32 @ffsll(i64 %0)            ; <i32> [#uses=1]
        ret i32 %2
}

; PR13028
define i32 @b() nounwind {
  %ffs = call i32 @ffsll(i64 0)
  ret i32 %ffs
; CHECK: @b
; CHECK-NEXT: ret i32 0
}
