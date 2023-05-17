; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s
; Check assembly printing of odd constants.

; CHECK: bigCst:
; CHECK-NEXT: .quad 12713950999227904
; CHECK-NEXT: .short 26220
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .zero 5
; CHECK-NEXT: .size bigCst, 16

@bigCst = internal constant i82 483673642326615442599424

define void @accessBig(ptr %storage) {
  %bigLoadedCst = load volatile i82, ptr @bigCst
  %tmp = add i82 %bigLoadedCst, 1
  store i82 %tmp, ptr %storage
  ret void
}
