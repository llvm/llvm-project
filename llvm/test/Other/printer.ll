; RUN: opt -passes='mem2reg,instcombine' -print-after-all -disable-output < %s 2>&1 | \
; RUN:   FileCheck %s --implicit-check-not='IR Dump'
define void @tester(){
  ret void
}

define void @foo(){
  ret void
}

; CHECK:      *** IR Dump After PromotePass
; CHECK-NEXT: define void @tester
; CHECK:      *** IR Dump After InstCombinePass
; CHECK-NEXT: define void @tester
; CHECK:      *** IR Dump After PromotePass
; CHECK-NEXT: define void @foo
; CHECK:      *** IR Dump After InstCombinePass
