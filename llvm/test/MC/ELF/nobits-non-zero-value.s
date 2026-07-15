# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:

## -filetype=asm does not check the error.
# RUN: llvm-mc -triple=x86_64 %s

.section .tbss,"aw",@nobits
  jmp foo

.bss
  addb %al,(%rax)

# CHECK: {{.*}}.s:[[#@LINE+1]]:11: warning: ignoring non-zero fill value in BSS section '.bss'
.align 4, 42

  .long 1

.section .bss0,"aw",%nobits
addb %al,(%rax)

.section data_fixup,"aw",%nobits
.quad foo

.section fill,"aw",%nobits
.fill b-a,1,1

.section org,"aw",%nobits
.org 1,1

.section ok,"aw",%nobits
.org 1
.fill 1
.fill b-a,1,0
.align 4, 0
.long 0

.text
a: nop
b:

## Location is not tracked for efficiency.
# CHECK: <unknown>:0: error: BSS section '.tbss' cannot have non-zero bytes
# CHECK: <unknown>:0: error: BSS section '.bss' cannot have non-zero bytes
# CHECK: <unknown>:0: error: BSS section 'data_fixup' cannot have fixups
# CHECK: <unknown>:0: error: BSS section 'fill' cannot have non-zero bytes
# CHECK: <unknown>:0: error: BSS section 'org' cannot have non-zero bytes
