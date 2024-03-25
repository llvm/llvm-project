; RUN: llc -O0 < %s -march=avr -mcpu avrxmega1 | FileCheck %s -check-prefix=XMEGA
; RUN: llc -O0 < %s -march=avr -mcpu avrxmega2 | FileCheck %s -check-prefix=XMEGA
; RUN: llc -O0 < %s -march=avr -mcpu avrxmega3 | FileCheck %s -check-prefix=XMEGA
; RUN: llc -O0 < %s -march=avr -mcpu avrxmega4 | FileCheck %s -check-prefix=XMEGA
; RUN: llc -O0 < %s -march=avr -mcpu avrxmega5 | FileCheck %s -check-prefix=XMEGA
; RUN: llc -O0 < %s -march=avr -mcpu avrxmega6 | FileCheck %s -check-prefix=XMEGA
; RUN: llc -O0 < %s -march=avr -mcpu avrxmega7 | FileCheck %s -check-prefix=XMEGA
; RUN: llc -O0 < %s -march=avr -mcpu avr2 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr25 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr3 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr31 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr35 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr4 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr5 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr51 | FileCheck %s -check-prefix=AVR
; RUN: llc -O0 < %s -march=avr -mcpu avr6 | FileCheck %s -check-prefix=AVR

define i8 @read8_low_io() {
; CHECK-LABEL: read8_low_io
; XMEGA: in r24, 8
; AVR: lds r24, 8
  %1 = load i8, ptr inttoptr (i16 8 to ptr)
  ret i8 %1
}

define i8 @read8_hi_io() {
; CHECK-LABEL: read8_hi_io
; XMEGA: in r24, 40
; AVR: in r24, 8
  %1 = load i8, ptr inttoptr (i16 40 to ptr)
  ret i8 %1
}

define i8 @read8_maybe_io() {
; CHECK-LABEL: read8_maybe_io
; XMEGA: lds r24, 80
; AVR: in r24, 48
  %1 = load i8, ptr inttoptr (i16 80 to ptr)
  ret i8 %1
}

define i8 @read8_not_io(){
; CHECK-LABEL: read8_not_io
; XMEGA: lds r24, 160
; AVR: lds r24, 160
  %1 = load i8, ptr inttoptr (i16 160 to ptr)
  ret i8 %1
}
