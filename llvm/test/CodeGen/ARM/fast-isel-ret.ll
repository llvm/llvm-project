; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s

; Sign-extend of i1 currently not supported by fast-isel
;define signext i1 @ret0(i1 signext %a) nounwind uwtable ssp {
;entry:
;  ret i1 %a
;}

define zeroext i1 @ret1(i1 signext %a) nounwind uwtable ssp {
entry:
; CHECK: ret1
; CHECK: and r0, r0, #1
; CHECK: bx lr
  ret i1 %a
}

define signext i8 @ret2(i8 signext %a) nounwind uwtable ssp {
entry:
; CHECK: ret2
; CHECK: sxtb r0, r0
; CHECK: bx lr
  ret i8 %a
}

define zeroext i8 @ret3(i8 signext %a) nounwind uwtable ssp {
entry:
; CHECK: ret3
; CHECK: uxtb r0, r0
; CHECK: bx lr
  ret i8 %a
}

define signext i16 @ret4(i16 signext %a) nounwind uwtable ssp {
entry:
; CHECK: ret4
; CHECK: sxth r0, r0
; CHECK: bx lr
  ret i16 %a
}

define zeroext i16 @ret5(i16 signext %a) nounwind uwtable ssp {
entry:
; CHECK: ret5
; CHECK: uxth r0, r0
; CHECK: bx lr
  ret i16 %a
}

define i16 @ret6(i16 %a) nounwind uwtable ssp {
entry:
; CHECK: ret6
; CHECK-NOT: uxth
; CHECK-NOT: sxth
; CHECK: bx lr
  ret i16 %a
}
