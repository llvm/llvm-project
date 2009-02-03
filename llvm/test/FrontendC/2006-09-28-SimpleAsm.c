// RUN: %llvmgcc %s -S -o - | grep {ext: xorl %eax, eax; movl}
// RUN: %llvmgcc %s -S -o - | grep {nonext: xorl %eax, %eax; mov}
// PR924

void bar() {
   // Extended asm
   asm volatile ("ext: xorl %%eax, eax; movl eax, fs; movl eax, gs  %%blah %= %% " : : "r"(1));
   // Non-extended asm.
   asm volatile ("nonext: xorl %eax, %eax; movl %eax, %fs; movl %eax, %gs  %%blah %= %% ");
}
