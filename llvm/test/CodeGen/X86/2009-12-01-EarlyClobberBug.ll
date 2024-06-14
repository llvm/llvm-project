; RUN: llc < %s -mtriple=x86_64-apple-darwin -no-integrated-as | FileCheck %s
; pr5391

define void @t() nounwind ssp {
entry:
; CHECK-LABEL: t:
; CHECK: movl %ecx, %eax
; CHECK: %eax = foo (%eax, %ecx)
  %b = alloca i32                                 ; <ptr> [#uses=2]
  %a = alloca i32                                 ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load i32, ptr %b, align 4                      ; <i32> [#uses=1]
  %1 = load i32, ptr %b, align 4                      ; <i32> [#uses=1]
  %asmtmp = call i32 asm "$0 = foo ($1, $2)", "=&{ax},%0,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp, ptr %a
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @t2() nounwind ssp {
entry:
; CHECK-LABEL: t2:
; CHECK: movl
; CHECK: [[D2:%e.x]] = foo
; CHECK: ([[D2]],
; CHECK-NOT: [[D2]]
; CHECK: )
  %b = alloca i32                                 ; <ptr> [#uses=2]
  %a = alloca i32                                 ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load i32, ptr %b, align 4                      ; <i32> [#uses=1]
  %1 = load i32, ptr %b, align 4                      ; <i32> [#uses=1]
  %asmtmp = call i32 asm "$0 = foo ($1, $2)", "=&r,%0,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp, ptr %a
  br label %return

return:                                           ; preds = %entry
  ret void
}
