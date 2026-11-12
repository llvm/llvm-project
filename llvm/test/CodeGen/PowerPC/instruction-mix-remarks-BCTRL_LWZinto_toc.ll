
; RUN: llc -verify-machineinstrs -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:   -pass-remarks-output=%t -pass-remarks=asm-printer -mcpu=pwr4 -o - %s
; RUN: FileCheck --input-file=%t %s

; CHECK:  - String:          "\n"
; CHECK:  - String:          "bctrl\n\tld 2, "
; CHECK:  - String:          ': '
; CHECK:  - INST_bctrl:      '1'
; CHECK:  - String:          "\n"


define void @callThroughPtrWithArgs(ptr nocapture) {
  tail call void %0(i32 signext 1, i16 zeroext 2, i64 3)
  ret void
}
