; Check the size of generated variable when no option is set
; RUN: opt -S %s -O2 -o - | FileCheck -check-prefix=CHECK-LONG %s
; CHECK-LONG: %{{[a-z]{4}[a-z]+}}

; Then check we correctly cap the size of newly generated non-global values name
; Force the size to be small so that the check works on release and debug build
; RUN: opt -S %s -O2 -o - -non-global-value-max-name-size=0 | FileCheck -check-prefix=CHECK-SHORT %s
; RUN: opt -S %s -O2 -o - -non-global-value-max-name-size=1 | FileCheck -check-prefix=CHECK-SHORT %s
; CHECK-SHORT-NOT: %{{[a-z][a-z]+}}

define i32 @f(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  %d = add i32 %c, %a
  %e = add i32 %d, %b
  ret i32 %e
}


