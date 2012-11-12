# RUN: llvm-mc -triple i386-unknown-unknown %s -I %p | FileCheck %s

.data
.incbin "incbin_abcd"

# CHECK: .ascii	 "abcd\n"
