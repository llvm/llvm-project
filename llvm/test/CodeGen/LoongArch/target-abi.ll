; RUN: llc --mtriple=loongarch32 --mattr=+d --target-abi=ilp32s < %s 2>&1 \
; RUN:   | FileCheck %s -DABI=ilp32s --check-prefixes=CHECK,WARNING
; RUN: llc --mtriple=loongarch32 --mattr=+d --target-abi=ilp32f < %s 2>&1 \
; RUN:   | FileCheck %s -DABI=ilp32f --check-prefixes=CHECK,WARNING
; RUN: llc --mtriple=loongarch32 --mattr=+d --target-abi=ilp32d < %s 2>&1 \
; RUN:   | FileCheck %s -DABI=ilp32d --check-prefixes=CHECK,WARNING
; RUN: llc --mtriple=loongarch64 --mattr=+d --target-abi=lp64f < %s 2>&1 \
; RUN:   | FileCheck %s -DABI=lp64f --check-prefixes=CHECK,WARNING

; RUN: llc --mtriple=loongarch64 --mattr=+d --target-abi=lp64s < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NO-WARNING
; RUN: llc --mtriple=loongarch64 --mattr=+d --target-abi=lp64d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NO-WARNING

;; Check if the ABI has been standardized; issue a warning if it hasn't.

; WARNING: warning: '[[ABI]]' has not been standardized

; NO-WARNING-NOT: warning

define void @nothing() nounwind {
; CHECK-LABEL: nothing:
; CHECK:       # %bb.0:
; CHECK-NEXT:  ret
  ret void
}
