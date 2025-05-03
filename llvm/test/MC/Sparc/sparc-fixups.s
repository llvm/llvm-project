! RUN: llvm-mc %s -triple=sparcv9 -filetype=obj | llvm-objdump -dr - | FileCheck %s
.text

! Check that fixups are correctly applied.

.set sym, 0xfedcba98

! CHECK:      sethi 0x3fb72e, %o0
! CHECK-NEXT: xor %o0, 0x298, %o0
! CHECK-NEXT: sethi 0x3b72ea, %o1
! CHECK-NEXT: xor %o0, 0x188, %o1
sethi %hi(sym), %o0
xor %o0, %lo(sym), %o0
sethi %hi(-0x12345678), %o1
xor %o0, %lo(-0x12345678), %o1

! CHECK:      sethi 0x3fb, %o0
! CHECK-NEXT: or %o0, 0x1cb, %o0
! CHECK-NEXT: ld [%o0+0xa98], %o0
sethi %h44(sym), %o0
or %o0, %m44(sym), %o0
ld [%o0 + %l44(sym)], %o0

! CHECK:      sethi 0x0, %o0
! CHECK-NEXT: sethi 0x3fb72e, %o0
! CHECK-NEXT: or %o0, 0x0, %o0
sethi %hh(sym), %o0
sethi %lm(sym), %o0
or %o0, %hm(sym), %o0

! CHECK:      sethi 0x48d1, %o0
! CHECK-NEXT: xor %o0, -0x168, %o0
sethi %hix(sym), %o0
xor %o0, %lox(sym), %o0
