; RUN: llc < %s -march=x86 -O0 -no-integrated-as | FileCheck %s
; PR7509
target triple = "i386-apple-darwin10"
%asmtype = type { i32, i8*, i32, i32 }

; Arguments 1 and 4 must be the same. No other output arguments may be
; allocated %eax.

; CHECK: InlineAsm Start
; CHECK: arg1 %[[A1:...]]
; CHECK-NOT: ax
; CHECK: arg4 %[[A1]]
; CHECK: InlineAsm End

define i32 @func(i8* %s) nounwind ssp {
entry:
  %0 = tail call %asmtype asm "arg0 $0\0A\09arg1 $1\0A\09arg2 $2\0A\09arg3 $3\0A\09arg4 $4", "={ax},=r,=r,=r,1,~{dirflag},~{fpsr},~{flags}"(i8* %s) nounwind, !srcloc !0 ; <%0> [#uses=1]
  %asmresult = extractvalue %asmtype %0, 0              ; <i64> [#uses=1]
  ret i32 %asmresult
}

!0 = !{i32 108}
