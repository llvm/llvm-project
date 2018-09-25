; RUN: opt -mem2reg -instcombine -print-after-all -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -passes='mem2reg,instcombine' -print-after-all -disable-output < %s 2>&1 | FileCheck %s
define void @tester(){
  ret void
}

define void @foo(){
  ret void
}

;CHECK-NOT: IR Dump After PassManager
;CHECK-NOT: IR Dump After ModuleToFunctionPassAdaptor
;
;CHECK:     *** IR Dump After {{Promote Memory to Register|PromotePass}}
;CHECK:     define void @tester
;CHECK-NOT: define void @foo
;CHECK:     *** IR Dump After {{Combine redundant instructions|InstCombinePass}}
;CHECK:     define void @tester
;CHECK-NOT: define void @foo
;CHECK:     *** IR Dump After {{Promote Memory to Register|PromotePass}}
;CHECK:     define void @foo
;CHECK-NOT: define void @tester
;CHECK:     *** IR Dump After {{Combine redundant instructions|InstCombinePass}}
;CHECK:     define void @foo
;CHECK-NOT: define void @tester
;CHECK:     *** IR Dump After {{Module Verifier|VerifierPass}}
;
;CHECK-NOT: IR Dump After Print Module IR
