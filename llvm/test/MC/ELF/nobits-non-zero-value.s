# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:

## -filetype=asm does not check the error.
# RUN: llvm-mc -triple=x86_64 %s

.section .tbss,"aw",@nobits
  jmp foo

.bss
  addb %al,(%rax)

# CHECK: {{.*}}.s:[[#@LINE+1]]:11: warning: ignoring non-zero fill value in SHT_NOBITS section '.bss'
.align 4, 42

.align 4, 0

  .long 1

.section .bss0,"aw",%nobits
addb %al,(%rax)

.section .bss1,"aw",%nobits
.quad foo

## Location is not tracked for efficiency.
# CHECK: <unknown>:0: error: SHT_NOBITS section '.tbss' cannot have non-zero bytes
# CHECK: <unknown>:0: error: SHT_NOBITS section '.bss' cannot have non-zero bytes
# CHECK: <unknown>:0: error: SHT_NOBITS section '.bss1' cannot have fixups
