; Test that external functions have jumptable entries emitted even if they are
; not address-taken when Cross-DSO CFI is used, but not otherwise.

; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/exported-funcs.yaml %s | FileCheck --check-prefixes=CHECK,CROSSDSO %s
; RUN: grep -v "llvm.module.flags" %s | opt -S -passes=lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/exported-funcs.yaml | FileCheck --check-prefixes=CHECK,NORMAL %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;;; Defined in the ThinLTO portion of the build (e.g. the summary)
; CROSSDSO: declare !type !1 !type !2 hidden void @external.cfi()
; NORMAL: declare !type !1 !type !2 void @external()
declare !type !1 !type !2 void @external()

; Don't emit jumptable entries for external declarations/non-external definitions
; CHECK-NOT: @external2
; CHECK-NOT: @internal

;;; Defined in the regular LTO portion of the build
; CROSSDSO: define hidden void @regularlto_external.cfi()
; NORMAL: define void @regularlto_external()
define void @regularlto_external() !type !1 !type !2 {
  ret void
}

; CHECK: define internal void @regularlto_internal()
define internal void @regularlto_internal() !type !1 !type !2 {
  ret void
}

!cfi.functions = !{!0, !3, !4}
!llvm.module.flags = !{!5}

!0 = !{!"external", i8 0, !1, !2}
!1 = !{i64 0, !"typeid1"}
!2 = !{i64 0, i64 1234}
!3 = !{!"external2", i8 1, !1, !2}
!4 = !{!"internal", i8 0, !1, !2}
!5 = !{i32 4, !"Cross-DSO CFI", i32 1}
