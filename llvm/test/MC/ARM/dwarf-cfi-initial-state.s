# RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o - \
# RUN:     | llvm-dwarfdump -v - | FileCheck %s

_proc:
.cfi_sections .debug_frame
.cfi_startproc
bx lr
.cfi_endproc

# CHECK: .debug_frame contents:
# CHECK: CIE
# CHECK-NOT: DW_CFA
# When llvm-dwarfdump -v prints the full info for the DW_CFA_def_cfa
# field, we can check that here too.
# CHECK: DW_CFA_def_cfa:
# The following 2 DW_CFA_nop instructions are "padding"
# CHECK: DW_CFA_nop:
# CHECK: DW_CFA_nop:
# CHECK-NOT: DW_CFA
# CHECK: FDE
