; RUN: opt -mem2reg -instcombine -print-after-all -S < %s 2>&1 | FileCheck %s
define void @tester(){
  ret void
}

define void @foo(){
  ret void
}

;CHECK: IR Dump After Promote Memory to Register
;CHECK: IR Dump After Combine redundant instructions
;CHECK: IR Dump After Module Verifier
;CHECK-NOT: IR Dump After Print Module IR
