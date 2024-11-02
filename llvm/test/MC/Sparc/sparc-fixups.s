! RUN: llvm-mc %s -arch=sparcv9 -filetype=obj | llvm-objdump -dr - | FileCheck %s
.text

! Check that fixups are correctly applied.

.set sym, 0xfedcba98

! CHECK: sethi 4175662, %o0
sethi %hi(sym), %o0
! CHECK: xor %o0, 664, %o0
xor %o0, %lo(sym), %o0

! CHECK: sethi 1019, %o0
sethi %h44(sym), %o0
! CHECK: or %o0, 459, %o0
or %o0, %m44(sym), %o0
! CHECK: ld [%o0+2712], %o0
ld [%o0 + %l44(sym)], %o0

! CHECK: sethi 0, %o0
sethi %hh(sym), %o0
! CHECK: sethi 4175662, %o0
sethi %lm(sym), %o0
! CHECK: or %o0, 0, %o0
or %o0, %hm(sym), %o0

! CHECK: sethi 18641, %o0
sethi %hix(sym), %o0
! CHECK: xor %o0, -360, %o0
xor %o0, %lox(sym), %o0
