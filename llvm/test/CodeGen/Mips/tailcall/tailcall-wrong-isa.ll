; RUN: llc -filetype=obj -mtriple=mipsel -relocation-model=pic -verify-machineinstrs -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=PIC32

; RUN: llc -filetype=obj -mtriple=mipsel -relocation-model=static -verify-machineinstrs -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=STATIC32

; RUN: llc -filetype=obj -mtriple=mips64el -relocation-model=pic -mcpu=mips64 -verify-machineinstrs -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=PIC64

; RUN: llc -filetype=obj -mtriple=mips64el -relocation-model=static -mcpu=mips64 -verify-machineinstrs -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=STATIC64

; RUN: llc -filetype=obj -mtriple=mipsel -relocation-model=pic -mattr=+micromips -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=PIC32MM

; RUN: llc -filetype=obj -mtriple=mipsel -relocation-model=static -mattr=+micromips -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=STATIC32MM

; RUN: llc -filetype=obj -mtriple=mipsel -relocation-model=pic -mcpu=mips32r6 -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s  --check-prefix=PIC32R6
; RUN: llc -filetype=obj -mtriple=mipsel -relocation-model=static -mcpu=mips32r6 -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=STATIC32R6

; RUN: llc -filetype=obj -mtriple=mips64el -relocation-model=pic -mcpu=mips64r6 -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=PIC64R6
; RUN: llc -filetype=obj -mtriple=mips64el -relocation-model=static -mcpu=mips64r6 -mips-tail-calls=1 < %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=STATIC64R6


define internal i8 @f2(i8) {
  ret i8 4
}

define i8 @f1(i8 signext %i) nounwind {
  %a = tail call i8 @f2(i8 %i)
  ret i8 %a
}

; ALL:        f1:
; PIC32:      {{[0-9a-z]}}: 08 00 20 03  jr $25
; STATIC32:   {{[0-9a-z]}}: 00 00 00 08  j 0

; PIC64:      {{[0-9a-z]+}}: 08 00 20 03  jr $25
; STATIC64:   {{[0-9]}}: 00 00 00 08  j 0

; PIC32MM:    {{[0-9a-z]+}}: b9 45 jrc $25
; STATIC32MM: {{[0-9a-z]}}: 00 d4 00 00 j 0

; PIC32R6:    {{[0-9a-z]}}: 00 00 19 d8  jrc $25
; STATIC32R6: {{[0-9a-z]}}: 00 00 00 08  j 0

; PIC64R6:    {{[0-9a-z]+}}: 00 00 19 d8  jrc $25
; STATIC64R6: {{[0-9]}}: 00 00 00 08  j 0
