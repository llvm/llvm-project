// By default, the jmp here does not need relaxation (so the 0xeb opdoce can be
// used).
// However, with -mc-relax-all passed to MC, all jumps are relaxed and we
// expect to see a different instruction.

// RUN: llvm-mc -filetype=obj -mc-relax-all -triple x86_64-pc-linux-gnu %s -o - \
// RUN:  | llvm-objdump -d - | FileCheck --check-prefix=RELAXALL %s

// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - \
// RUN:  | llvm-objdump -d - | FileCheck %s

.text
foo:
  mov %rax, %rax
  jmp foo

// RELAXALL:    3:  e9
// CHECK:       3:  eb

