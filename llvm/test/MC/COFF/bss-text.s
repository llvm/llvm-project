# RUN: not llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## -filetype=asm does not check the error.
# RUN: llvm-mc -triple=x86_64-pc-win32 %s

.bss
# CHECK: <unknown>:0: error: BSS section '.bss' cannot have non-zero bytes
  addb %bl,(%rax)

.section uninitialized,"b"
# CHECK: <unknown>:0: error: BSS section 'uninitialized' cannot have non-zero bytes
  jmp foo

.section bss0,"b"
  addb %al,(%rax)
