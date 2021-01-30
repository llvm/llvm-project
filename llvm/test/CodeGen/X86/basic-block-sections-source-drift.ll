; RUN: echo "!foo" > %t.order.txt
; RUN: llc < %s -mtriple=x86_64-pc-linux -basic-block-sections=%t.order.txt  | FileCheck --check-prefix=SOURCE-DRIFT %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -basic-block-sections=%t.order.txt -bbsections-detect-source-drift=false | FileCheck --check-prefix=HASH-CHECK-DISABLED %s

define dso_local i32 @foo(i1 zeroext %0, i1 zeroext %1)  !annotation !1 {
  br i1 %0, label %5, label %3

3:                                                ; preds = %2
  %4 = select i1 %1, i32 2, i32 0
  ret i32 %4

5:                                                ; preds = %2
  ret i32 1
}

!1 = !{!"instr_prof_hash_mismatch"}

; SOURCE-DRIFT-NOT: .section .text
; HASH-CHECK-DISABLED: .section .text
