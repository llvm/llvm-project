# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck %s --check-prefix=RELAX

## Ensure that the eh_frame records the symbolic difference with
## the R_LARCH_32_PCREL relocation.

func:
 .cfi_startproc
  ret
 .cfi_endproc

# CHECK:   Section (4) .rela.eh_frame {
# CHECK-NEXT:   0x1C R_LARCH_32_PCREL .text 0x0
# CHECK-NEXT: }
# RELAX:   Section ({{.*}}) .rela.eh_frame {
# RELAX-NEXT:   0x1C R_LARCH_32_PCREL .L{{.*}} 0x0
# RELAX-NEXT: }
