; RUN: not llc -mtriple=armv7-windows-itanium -mcpu=cortex-a9 -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s -check-prefixes=CHECK-OPTIONS,CHECK-FEATURE

; RUN: not llc -mtriple=thumb-unknown-linux -mcpu=cortex-m0 -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s -check-prefix=CHECK-FEATURE

define void @foo() {
entry:
  ret void
}

; CHECK-OPTIONS: Function 'foo' uses ARM instructions, but the target does not support ARM mode execution.

define void @no_thumb_mode_feature() #0 {
entry:
  ret void
}

; CHECK-FEATURE: Function 'no_thumb_mode_feature' uses ARM instructions, but the target does not support ARM mode execution.

attributes #0 = { "target-features"="-thumb-mode" }
