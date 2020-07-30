; Just because a function is named like an intrinsic does not mean we should skip it's attributes.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-ALL: declare i32 @llvm.not.really.an.intrinsic(i32, i32) #0
declare i32 @llvm.not.really.an.intrinsic(i32, i32) #0

define i32 @t(i32 %a) {
; CHECK-ALL-LABEL: @t(

; CHECK-INTERESTINGNESS: %r =
; CHECK-INTERESTINGNESS-SAME: call
; CHECK-INTERESTINGNESS-SAME: "arg0"
; CHECK-INTERESTINGNESS-SAME: i32 @llvm.not.really.an.intrinsic(i32
; CHECK-INTERESTINGNESS-SAME: "arg3"
; CHECK-INTERESTINGNESS-SAME: %a
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: %a
; CHECK-INTERESTINGNESS-SAME: #1

; CHECK-FINAL: %r = call "arg0" i32 @llvm.not.really.an.intrinsic(i32 "arg3" %a, i32 %a) #1
; CHECK-ALL: ret i32 %r

  %r = call "arg0" "arg1" i32 @llvm.not.really.an.intrinsic(i32 "arg2" "arg3" %a, i32 %a) "arg4" "arg5"
  ret i32 %r
}

; CHECK-INTERESTINGNESS: attributes #0 = {
; CHECK-INTERESTINGNESS-SAME: "arg6"

; CHECK-INTERESTINGNESS: attributes #1 = {
; CHECK-INTERESTINGNESS-SAME: "arg4"

; CHECK-FINAL: attributes #0 = { "arg6" }
; CHECK-FINAL: attributes #1 = { "arg4" }

; CHECK-ALL-NOT: attributes #

attributes #0 = { "arg6" "arg7" }
