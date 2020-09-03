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

# PR18599
.macro macro_a
 .macro macro_b
  .byte 10
  .macro macro_c
  .endm

  macro_c
  .purgem macro_c
 .endm

 macro_b
.endm

# CHECK: .byte 10
# CHECK: .byte 10
macro_a
macro_b
