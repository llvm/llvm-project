; RUN: llc <%s --mtriple s390x-ibm-zos --filetype=obj -o - | \
; RUN:   od -Ax -tx1 -v | FileCheck --ignore-case %s

source_filename = "test.ll"

declare void @other(...)

define void @me() {
entry:
  tail call void @other()
  ret void
}

; Header record:
;  03 is prefix byte
;  f. is header type
;  .0 is flag
;  00 is version
; The 1 at offset 0x33 is the architecture level.
; CHECK: 000000 03 f0 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000010 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000030 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

; ESD record, type SD.
;  03 is prefix byte
;  0. is header type
;  .0 is flag
;  00 is version
;  00 is type = SD
; The 01 at offset 0x57 is the id of the symbol.
; The 60 at offset 0x89 is the tasking behavior.
; The 01 at offset 0x91 is the binding scope.
; The name begins at offset 0x97, and is test#C.
; CHECK: 0000050 03 00 00 00 00 00 00 01 00 00 00 00 00 00 00 00
; CHECK: 0000060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0000070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0000080 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60
; CHECK: 0000090 00 01 00 00 00 00 00 06 a3 85 a2 a3 7b c3 00 00

; ESD record, type ED.
; The name is C_WSA64.
; CHECK: 00000a0 03 00 00 01 00 00 00 02 00 00 00 01 00 00 00 00
; CHECK: 00000b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00000c0 00 00 00 00 00 00 00 00 03 81 00 00 00 00 00 00
; CHECK: 00000d0 00 00 00 00 00 00 00 00 00 00 00 00 04 04 01 01
; CHECK: 00000e0 00 40 04 00 00 00 00 07 c3 6d e6 e2 c1 f6 f4 00

; ESD record, type PR.
; The name is test#S.
; CHECK: 00000f0 03 00 00 03 00 00 00 03 00 00 00 02 00 00 00 00
; CHECK: 0000100 00 00 00 00 00 00 00 00 00 00 00 10 00 00 00 00
; CHECK: 0000110 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
; CHECK: 0000120 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 01
; CHECK: 0000130 00 01 24 00 00 00 00 06 a3 85 a2 a3 7b e2 00 00

; ESD record, type ED.
; The name is C_CODE64.
; The regular expression matches the low byte of the length.
; CHECK: 0000140 03 00 00 01 00 00 00 04 00 00 00 01 00 00 00 00
; CHECK: 0000150 00 00 00 00 00 00 00 00 00 00 00 {{..}} 00 00 00 00
; CHECK: 0000160 00 00 00 00 00 00 00 00 01 80 00 00 00 00 00 00
; CHECK: 0000170 00 00 00 00 00 00 00 00 00 00 00 00 04 04 00 0a
; CHECK: 0000180 00 00 03 00 00 00 00 08 c3 6d c3 d6 c4 c5 f6 f4

; ESD record, type LD.
; The name is test#C.
; CHECK: 00002d0 03 00 00 02 00 00 00 08 00 00 00 04 00 00 00 00
; CHECK: 00002e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00002f0 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 03
; CHECK: 0000300 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 02
; CHECK: 0000310 00 01 20 00 00 00 00 06 a3 85 a2 a3 7b c3 00 00
