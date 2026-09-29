# REQUIRES: x86-registered-target

## Verify llvm-bitcode-strip removes sections from object files
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t
# RUN: llvm-bitcode-strip -r %t -o %t2
# RUN: llvm-readobj --macho-segment --sections %t2 | FileCheck --implicit-check-not=__LLVM %s

# CHECK:      Name: __text
# CHECK-NEXT: Segment: __TEXT

.section __LLVM,__bundle
  .asciz "test"

.section __LLVM,__bitcode
  .asciz "test"

.section __LLVM,__cmdline
  .asciz "test"

.section __LLVM,__swift_cmdline
  .asciz "test"

.section __LLVM,__asm
  .asciz "test"

.text
.globl _main
_main:
  ret
