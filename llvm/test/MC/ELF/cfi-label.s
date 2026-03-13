# RUN: llvm-mc -triple x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -sX %t | FileCheck %s --check-prefix=SYMTAB
# RUN: llvm-dwarfdump --eh-frame %t | FileCheck %s

# RUN: not llvm-mc -filetype=obj -triple=x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ASM:      nop
# ASM-NEXT: .cfi_label cfi1
# ASM-NEXT: .cfi_escape 0x00
# ASM:      .globl cfi2
# ASM-NEXT: .cfi_label cfi2
# ASM:      nop
# ASM-NEXT: .cfi_label .Lcfi3

# SYMTAB:      000000000000002b     0 NOTYPE  LOCAL  DEFAULT     3 (.eh_frame) cfi1
# SYMTAB:      000000000000002d     0 NOTYPE  GLOBAL DEFAULT     3 (.eh_frame) cfi2
# SYMTAB-NOT:  {{.}}

# CHECK:       DW_CFA_remember_state:
# CHECK-NEXT:  DW_CFA_advance_loc: 1 to 0x1
# CHECK-NEXT:  DW_CFA_nop:
# CHECK-NEXT:  DW_CFA_advance_loc: 1 to 0x2
# CHECK-NEXT:  DW_CFA_nop:
# CHECK-NEXT:  DW_CFA_nop:
# CHECK-NEXT:  DW_CFA_advance_loc: 1 to 0x3
# CHECK-NEXT:  DW_CFA_nop:
# CHECK-NEXT:  DW_CFA_nop:
# CHECK-NEXT:  DW_CFA_nop:
# CHECK-NEXT:  DW_CFA_restore_state:

.globl foo
foo:
.cfi_startproc
.cfi_remember_state
nop
.cfi_label cfi1
.cfi_escape 0
nop
.globl cfi2
.cfi_label cfi2
.cfi_escape 0, 0
nop
.cfi_label .Lcfi3
.cfi_escape 0, 0, 0
.cfi_restore_state
ret

# ERR: [[#@LINE+10]]:1: error: this directive must appear between .cfi_startproc and .cfi_endproc directives
.ifdef ERR
# ERR: [[#@LINE+1]]:12: error: symbol 'foo' is already defined
.cfi_label foo
# ERR: [[#@LINE+1]]:12: error: symbol '.Lcfi3' is already defined
.cfi_label .Lcfi3
.endif
.cfi_endproc

.ifdef ERR
.cfi_label after_endproc
.endif
