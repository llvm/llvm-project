; RUN: llc -mtriple=x86_64 -relocation-model=pic < %s | FileCheck %s

;; With a module flag SemanticInterposition=0, infer dso_local flags even if PIC.
;; Local aliases will be generated for applicable variables and functions.

@var = global i32 0, align 4

@ifunc = ifunc i32 (), bitcast (i32 ()* ()* @ifunc_resolver to i32 ()*)

define i32 @ifunc_impl() {
entry:
  ret i32 0
}

define i32 ()* @ifunc_resolver() {
entry:
  ret i32 ()* @ifunc_impl
}

declare i32 @external()

define i32 @func() {
  ret i32 0
}

;; Don't set dso_local on declarations or ifuncs.
define i32 @foo() {
; CHECK: movl .Lvar$local(%rip), %ebp
; CHECK: callq external@PLT
; CHECK: callq ifunc@PLT
; CHECK: callq .Lfunc$local{{$}}
entry:
  %0 = load i32, i32* @var, align 4
  %call = tail call i32 @external()
  %add = add nsw i32 %call, %0
  %call1 = tail call i32 @ifunc()
  %add2 = add nsw i32 %add, %call1
  %call2 = tail call i32 @func()
  %add3 = add nsw i32 %add, %call2
  ret i32 %add3
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"SemanticInterposition", i32 0}
!1 = !{i32 7, !"PIC Level", i32 2}
