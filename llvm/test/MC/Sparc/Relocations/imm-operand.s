## Test relocations for symbolic immediate operands of sethi, unimp, Tcc, movr
## and movcc. A value that is only known after encoding (a label difference or
## a later .set) is resolved into the field.
# RUN: llvm-mc -triple=sparc %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple=sparcv9 %s --defsym V9=1 | FileCheck %s --check-prefixes=ASM,ASM64
# RUN: llvm-mc -triple=sparc -filetype=obj %s -o %t
# RUN: llvm-objdump -dr %t | FileCheck %s --check-prefix=OBJ
# RUN: llvm-mc -triple=sparcv9 -filetype=obj %s --defsym V9=1 -o %t.64
# RUN: llvm-objdump -dr %t.64 | FileCheck %s --check-prefixes=OBJ,OBJ64

# ASM:      sethi sym, %g1
# ASM-NEXT: unimp sym
# ASM-NEXT: ta sym+4
# OBJ:      sethi 0x0, %g1
# OBJ-NEXT:   R_SPARC_22 sym{{$}}
# OBJ-NEXT: unimp 0x0
# OBJ-NEXT:   R_SPARC_22 sym{{$}}
# OBJ-NEXT: ta {{(%icc, )?}}0x0
# OBJ-NEXT:   R_SPARC_7 sym+0x4
sethi sym, %g1
unimp sym
ta sym + 4

.ifdef V9
# ASM64:      tne %xcc, sym
# ASM64-NEXT: movrz %o0, sym, %o1
# ASM64-NEXT: movrnz %o0, sym+4, %o1
# ASM64-NEXT: movne %icc, sym, %o1
# ASM64-NEXT: movne %xcc, sym, %o1
# ASM64-NEXT: movne %fcc1, sym, %o1
# ASM64-NEXT: movrz %o0, %lo(sym), %o1
# OBJ64:      tne %xcc, 0x0
# OBJ64-NEXT:   R_SPARC_7 sym{{$}}
# OBJ64-NEXT: movrz %o0, 0x0, %o1
# OBJ64-NEXT:   R_SPARC_10 sym{{$}}
# OBJ64-NEXT: movrnz %o0, 0x0, %o1
# OBJ64-NEXT:   R_SPARC_10 sym+0x4
# OBJ64-NEXT: movne %icc, 0x0, %o1
# OBJ64-NEXT:   R_SPARC_11 sym{{$}}
# OBJ64-NEXT: movne %xcc, 0x0, %o1
# OBJ64-NEXT:   R_SPARC_11 sym{{$}}
# OBJ64-NEXT: movne %fcc1, 0x0, %o1
# OBJ64-NEXT:   R_SPARC_11 sym{{$}}
# OBJ64-NEXT: movrz %o0, 0x0, %o1
# OBJ64-NEXT:   R_SPARC_LO10 sym{{$}}
tne %xcc, sym
movrz %o0, sym, %o1
movrnz %o0, sym + 4, %o1
movne %icc, sym, %o1
movne %xcc, sym, %o1
movne %fcc1, sym, %o1
movrz %o0, %lo(sym), %o1
.endif

# OBJ-NEXT:  sethi 0x3fffff, %g1
# OBJ-NEXT:  unimp 0x10
# OBJ-NEXT:  ta {{(%icc, )?}}0x10
sethi x, %g1
unimp b - a
ta b - a

.ifdef V9
# OBJ64-NEXT: tne %xcc, 0x10
# OBJ64-NEXT: movrz %o0, 0x3fc, %o1
# OBJ64-NEXT: movne %xcc, 0x10, %o1
# OBJ64-NEXT: movne %icc, 0x7fc, %o1
tne %xcc, b - a
movrz %o0, y, %o1
movne %xcc, b - a, %o1
movne %icc, y, %o1
.endif

.data
a:
.skip 16
b:

.set x, 0x3fffff
.set y, -4
