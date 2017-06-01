//RUN:	not llvm-mc --disassemble -triple=x86_64 --output-asm-variant=2 %s -o - 2>&1 | FileCheck %s

//CHECK: error: unable to create instruction printer for target triple 'x86_64' with assembly variant 2.
