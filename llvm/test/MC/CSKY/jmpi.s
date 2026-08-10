# RUN: llvm-mc -filetype=obj -triple=csky  < %s \
# RUN:     | llvm-objdump  --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s


.text

LABEL:
  bkpt
  jmpi LABEL
  bkpt


# CHECK:        0:      	bkpt
# CHECK-NEXT:   2:      	br32	0x0
# CHECK-NEXT:   6:      	bkpt


# CHECK:        8:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:   			    00000008:  R_CKCORE_ADDR32	.text

