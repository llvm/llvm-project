; RUN: llc  %s -mtriple=arm-linux-gnueabi -filetype=obj -o - | \
; RUN:    elf-dump --dump-section-data | FileCheck  -check-prefix=BASIC %s 
; RUN: llc  %s -mtriple=armv7-linux-gnueabi -march=arm -mcpu=cortex-a8 \
; RUN:    -mattr=-neon,-vfp3,+vfp2 \
; RUN:    -arm-reserve-r9 -filetype=obj -o - | \
; RUN:    elf-dump --dump-section-data | FileCheck  -check-prefix=CORTEXA8 %s


; This tests that the extpected ARM attributes are emitted.
;
; BASIC:        .ARM.attributes
; BASIC-NEXT:         0x70000003
; BASIC-NEXT:         0x00000000
; BASIC-NEXT:         0x00000000
; BASIC-NEXT:         0x0000003c
; BASIC-NEXT:         0x00000022
; BASIC-NEXT:         0x00000000
; BASIC-NEXT:         0x00000000
; BASIC-NEXT:         0x00000001
; BASIC-NEXT:         0x00000000
; BASIC-NEXT:         '41210000 00616561 62690001 17000000 060a0741 08010902 14011501 17031801 1901'

; CORTEXA8:        .ARM.attributes
; CORTEXA8-NEXT:         0x70000003
; CORTEXA8-NEXT:         0x00000000
; CORTEXA8-NEXT:         0x00000000
; CORTEXA8-NEXT:         0x0000003c
; CORTEXA8-NEXT:         0x0000002f
; CORTEXA8-NEXT:         0x00000000
; CORTEXA8-NEXT:         0x00000000
; CORTEXA8-NEXT:         0x00000001
; CORTEXA8-NEXT:         0x00000000
; CORTEXA8-NEXT:         '412e0000 00616561 62690001 24000000 05434f52 5445582d 41380006 0a074108 0109020a 02140115 01170318 011901'

define i32 @f(i64 %z) {
       ret i32 0
}
