; Test that GVN is skipped when function has zero entry count in PGO
; RUN: opt -passes='gvn' -gvn-skip-zero-entry-count=true -S < %s | FileCheck %s

; Function with ZERO entry count - GVN should skip this function
; The redundant computation should remain because GVN doesn't run
; CHECK-LABEL: @zero_freq_function(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = add i32 %x, 1
; CHECK-NEXT:   %b = add i32 %a, 2
; CHECK-NEXT:   %c = add i32 %a, 2
; CHECK-NEXT:   %result = add i32 %b, %c
; CHECK-NEXT:   ret i32 %result
define i32 @zero_freq_function(i32 %x) !prof !0 {
entry:
  %a = add i32 %x, 1
  %b = add i32 %a, 2
  %c = add i32 %a, 2    ; Redundant - but GVN should not  optimize due to zero freq
  %result = add i32 %b, %c
  ret i32 %result
}

; Function with NON-ZERO entry count - GVN should run normally
; The redundant computation should be eliminated by GVN
; CHECK-LABEL: @nonzero_freq_function(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = add i32 %x, 1
; CHECK-NEXT:   %b = add i32 %a, 2
; CHECK-NEXT:   %result = add i32 %b, %b
; CHECK-NEXT:   ret i32 %result
define i32 @nonzero_freq_function(i32 %x) !prof !1 {
entry:
  %a = add i32 %x, 1
  %b = add i32 %a, 2
  %c = add i32 %a, 2    ; Redundant - GVN optimizes this
  %result = add i32 %b, %c
  ret i32 %result
}

!0 = !{!"function_entry_count", i64 0}      ; Zero frequency
!1 = !{!"function_entry_count", i64 1000}   ; Non-zero frequency

