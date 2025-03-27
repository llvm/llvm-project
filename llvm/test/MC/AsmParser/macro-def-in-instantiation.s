# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s
# RUN: llvm-mc -triple=x86_64-apple-darwin10 %s | FileCheck %s

.macro make_macro a, b, c ,d ,e, f
\a \b \c
\d \e
\f
.endm
make_macro .macro,mybyte,a,.byte,\a,.endm
# CHECK: .byte 42
mybyte 42
