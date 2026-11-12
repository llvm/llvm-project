; RUN: llc -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s

; When any x87 register is touched, IPRA must treat the entire x87 FP stack as
; clobbered (FP0-FP7, ST0-ST7), not just the individually used reg.


; CHECK-LABEL: caller Clobbered Registers:
; CHECK-SAME: $fp0
; CHECK-SAME: $fp7
define void @caller() #0 {
  call void @uses_x87()
  call void @no_x87()
  ret void
}


; CHECK-LABEL: no_x87 Clobbered Registers:
; CHECK-NOT: $fp0
; CHECK-NOT: $st0
define void @no_x87() #0 {
  ret void
}

; CHECK-LABEL: uses_x87 Clobbered Registers:
; CHECK-SAME: $fp0
; CHECK-SAME: $fp7
define void @uses_x87() #0 {
  call void asm sideeffect "fld1", "~{fp0}"()
  ret void
}

@llvm.used = appending global [3 x ptr] [ptr @uses_x87, ptr @no_x87, ptr @caller]

attributes #0 = { nounwind }
