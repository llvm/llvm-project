# RUN: llvm-mc -triple x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

# ASM:   .section .linkorder,"ao",@progbits,0
# CHECK: Name       Type     {{.*}} Flg Lk
# CHECK: .linkorder PROGBITS {{.*}}  AL  0
.section .linkorder,"ao",@progbits,0
