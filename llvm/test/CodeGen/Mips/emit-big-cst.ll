; RUN: llc -march=mips < %s | FileCheck %s --check-prefix=BE
; RUN: llc -march=mipsel < %s | FileCheck %s --check-prefix=LE
; Check assembly printing of odd constants.

; BE-LABEL: bigCst:
; BE-NEXT: .8byte 28829195638097253
; BE-NEXT: .2byte 46
; BE-NEXT: .byte 0
; BE-NEXT: .space 5
; BE-NEXT: .size bigCst, 16

; LE-LABEL: bigCst:
; LE-NEXT: .8byte 12713950999227904
; LE-NEXT: .2byte 26220
; LE-NEXT: .byte 0
; LE-NEXT: .space 5
; LE-NEXT: .size bigCst, 16

; BE-LABEL: smallCst:
; BE-NEXT: .2byte 4386
; BE-NEXT: .byte 51
; BE-NEXT: .space 1
; BE-NEXT: .size smallCst, 4

; LE-LABEL: smallCst:
; LE-NEXT: .2byte 8755
; LE-NEXT: .byte 17
; LE-NEXT: .space 1
; LE-NEXT: .size smallCst, 4

@bigCst = internal constant i82 483673642326615442599424

define void @accessBig(i64* %storage) {
  %addr = bitcast i64* %storage to i82*
  %bigLoadedCst = load volatile i82, i82* @bigCst
  %tmp = add i82 %bigLoadedCst, 1
  store i82 %tmp, i82* %addr
  ret void
}

@smallCst = internal constant i24 1122867
