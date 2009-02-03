; RUN: llvm-as < %s | llc -march=x86-64 -fast | grep movslq
; RUN: llvm-as < %s | llc -march=x86 -fast
; PR3181

; GEP indices are interpreted as signed integers, so they
; should be sign-extended to 64 bits on 64-bit targets.

define i32 @foo(i32 %t3, i32* %t1) nounwind {
       %t9 = getelementptr i32* %t1, i32 %t3           ; <i32*> [#uses=1]
       %t15 = load i32* %t9            ; <i32> [#uses=1]
       ret i32 %t15
}
define i32 @bar(i64 %t3, i32* %t1) nounwind {
       %t9 = getelementptr i32* %t1, i64 %t3           ; <i32*> [#uses=1]
       %t15 = load i32* %t9            ; <i32> [#uses=1]
       ret i32 %t15
}
