# RUN: not llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o /dev/null 2>&1 | FileCheck %s

## -filetype=asm does not check the error.
# RUN: llvm-mc -triple=x86_64-pc-win32 %s

.section uninitialized,"b"
# MCRelaxableFragment
# CHECK: {{.*}}.s:[[#@LINE+1]]:3: error: IMAGE_SCN_CNT_UNINITIALIZED_DATA section 'uninitialized' cannot have instructions
  jmp foo

.bss
# CHECK: {{.*}}.s:[[#@LINE+1]]:3: error: IMAGE_SCN_CNT_UNINITIALIZED_DATA section '.bss' cannot have instructions
  addb %al,(%rax)
