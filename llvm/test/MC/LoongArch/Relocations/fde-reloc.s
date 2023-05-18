# RUN: llvm-mc --filetype=obj --triple=loongarch64 < %s \
# RUN:     | llvm-readobj -r - | FileCheck %s

## Ensure that the eh_frame records the symbolic difference with
## the R_LARCH_32_PCREL relocation.

func:
 .cfi_startproc
  ret
 .cfi_endproc

# CHECK:   Section (4) .rela.eh_frame {
# CHECK-NEXT:   0x1C R_LARCH_32_PCREL .text 0x0
# CHECK-NEXT: }
