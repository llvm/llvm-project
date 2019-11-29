; RUN: opt -mem2reg -instcombine -print-after-all -disable-output < %s 2>&1 | \
; RUN:   FileCheck --check-prefixes=CHECK,OLDPM %s --implicit-check-not='IR Dump'
; RUN: opt -passes='mem2reg,instcombine' -print-after-all -disable-output < %s 2>&1 | \
; RUN:   FileCheck --check-prefixes=CHECK,NEWPM %s --implicit-check-not='IR Dump'
define void @tester(){
  ret void
}

define void @foo(){
  ret void
}

; NEWPM:      *** IR Dump After VerifierPass
; CHECK:      *** IR Dump After {{Promote Memory to Register|PromotePass}}
; CHECK-NEXT: define void @tester
; CHECK:      *** IR Dump After {{Combine redundant instructions|InstCombinePass}}
; CHECK-NEXT: define void @tester
; OLDPM:      *** IR Dump After Module Verifier
; CHECK:      *** IR Dump After {{Promote Memory to Register|PromotePass}}
; CHECK-NEXT: define void @foo
; CHECK:      *** IR Dump After {{Combine redundant instructions|InstCombinePass}}
; CHECK-NEXT: define void @foo
; CHECK:      *** IR Dump After {{Module Verifier|VerifierPass}}
