# RUN: llvm-mc -filetype=obj -triple x86_64 %s | llvm-readelf -S - | FileCheck %s

# CHECK:      Section Headers:
# CHECK-NOT:  .eh_frame
# CHECK-NOT:  .debug_frame

.cfi_sections

## .cfi_startproc and .cfi_endproc are ignored.
.cfi_startproc
nop
.cfi_endproc
