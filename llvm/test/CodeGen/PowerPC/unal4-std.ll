; RUN: llc < %s -mcpu=pwr7 -mattr=-vsx| FileCheck %s
; RUN: llc < %s -mcpu=pwr7 -mattr=+vsx | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @copy_to_conceal(<8 x i16>* %inp) #0 {
entry:
  store <8 x i16> zeroinitializer, <8 x i16>* %inp, align 2
  br label %if.end210

if.end210:                                        ; preds = %entry
  ret void

; This will generate two align-1 i64 stores. Make sure that they are
; indexed stores and not in r+i form (which require the offset to be
; a multiple of 4).
; CHECK: @copy_to_conceal
; CHECK: stdx {{[0-9]+}}, 0,

; CHECK-VSX: @copy_to_conceal
; CHECK-VSX: stxvw4x {{[0-9]+}}, 0,
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
