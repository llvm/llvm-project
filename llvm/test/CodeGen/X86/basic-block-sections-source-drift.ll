; RUN: echo "v1" > %t
; RUN: echo "f foo" >> %t
; RUN: echo "c 0" >> %t
; RUN: llc < %s -mtriple=x86_64-pc-linux -basic-block-sections=%t  | FileCheck --check-prefix=SOURCE-DRIFT %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -basic-block-sections=%t -bbsections-detect-source-drift=false | FileCheck --check-prefix=HASH-CHECK-DISABLED %s

define dso_local i32 @foo(i1 zeroext %0, i1 zeroext %1)  !annotation !1 {
  br i1 %0, label %5, label %3

3:                                                ; preds = %2
  %4 = select i1 %1, i32 2, i32 0
  ret i32 %4

5:                                                ; preds = %2
  ret i32 1
}

!1 = !{!"instr_prof_hash_mismatch"}

; SOURCE-DRIFT-NOT: .section .text{{.*}}.foo
; HASH-CHECK-DISABLED: .section .text.hot.foo
