# RUN: llvm-mc %s -triple=sparc -filetype=obj -o - | llvm-readobj -r - | FileCheck %s --check-prefix=V8
# RUN: llvm-mc %s --defsym V9=1 -triple=sparcv9 -filetype=obj -o - | llvm-readobj -r - | FileCheck %s --check-prefix=V9

## All three relocations of a GOTDATA_OP sequence against a local symbol
## reference the symbol itself rather than its section, as GNU as does.

# V8:      Section ({{[0-9]+}}) .rela.text {
# V8-NEXT:   0x0 R_SPARC_GOTDATA_OP_HIX22 local 0x0
# V8-NEXT:   0x4 R_SPARC_GOTDATA_OP_LOX10 local 0x0
# V8-NEXT:   0x8 R_SPARC_GOTDATA_OP local 0x0
# V8-NEXT: }

# V9:      Section ({{[0-9]+}}) .rela.text {
# V9-NEXT:   0x0 R_SPARC_GOTDATA_OP_HIX22 local 0x0
# V9-NEXT:   0x4 R_SPARC_GOTDATA_OP_LOX10 local 0x0
# V9-NEXT:   0x8 R_SPARC_GOTDATA_OP local 0x0
# V9-NEXT: }

  sethi %gdop_hix22(local), %l1
  xor %l1, %gdop_lox10(local), %l1
.ifdef V9
  ldx [%l7 + %l1], %l2, %gdop(local)
.else
  ld [%l7 + %l1], %l2, %gdop(local)
.endif

  .data
  .word 0
local:
  .word 0
