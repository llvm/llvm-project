# RUN: llvm-mc -filetype=obj --triple=loongarch64 %s | llvm-readobj -r - | FileCheck %s

# CHECK: R_LARCH_TLS_DTPREL32
.dtprelword x

# CHECK: R_LARCH_TLS_DTPREL64
.dtpreldword x
