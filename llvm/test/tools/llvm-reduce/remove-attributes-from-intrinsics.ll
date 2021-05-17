; We can't actually put attributes on intrinsic declarations, only on call sites.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

define i32 @t(i32 %a) {
; CHECK-ALL-LABEL: @t(

; CHECK-INTERESTINGNESS: %r =
; CHECK-INTERESTINGNESS-SAME: call
; CHECK-INTERESTINGNESS-SAME: "arg0"
; CHECK-INTERESTINGNESS-SAME: i32 @llvm.uadd.sat.i32(i32
; CHECK-INTERESTINGNESS-SAME: "arg3"
; CHECK-INTERESTINGNESS-SAME: %a
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: %a
; CHECK-INTERESTINGNESS-SAME: #1

; CHECK-FINAL: %r = call "arg0" i32 @llvm.uadd.sat.i32(i32 "arg3" %a, i32 %a) #1
; CHECK-ALL: ret i32 %r

  %r = call "arg0" "arg1" i32 @llvm.uadd.sat.i32(i32 "arg2" "arg3" %a, i32 %a) "arg4" "arg5"
  ret i32 %r
}

; CHECK-ALL: declare i32 @llvm.uadd.sat.i32(i32, i32) #0
declare i32 @llvm.uadd.sat.i32(i32, i32) #0

; CHECK-ALL: attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

; CHECK-INTERESTINGNESS: attributes #1 = {
; CHECK-INTERESTINGNESS-SAME: "arg4"

; CHECK-FINAL: attributes #1 = { "arg4" }

; CHECK-ALL-NOT: attributes #

attributes #0 = { "arg6" "arg7" }
