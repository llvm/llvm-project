; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; Make sure that speculatable is not allowed on a call site if the
; declaration is not also speculatable.

declare i32 @not_speculatable()

; CHECK: speculatable attribute may not apply to call sites
; CHECK-NEXT: %ret = call i32 @not_speculatable() #0
define i32 @call_not_speculatable() {
  %ret = call i32 @not_speculatable() #0
  ret i32 %ret
}

@gv = internal unnamed_addr constant i32 0

; CHECK: speculatable attribute may not apply to call sites
; CHECK-NEXT: %ret = call float @gv() #0
define float @call_bitcast_speculatable() {
  %ret = call float @gv() #0
  ret float %ret
}

attributes #0 = { speculatable }
