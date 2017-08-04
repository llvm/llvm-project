@ RUN: llvm-mc < %s -triple armv7m-elf -filetype=obj | llvm-objdump -triple=thumb -d - | FileCheck %s

.arch armv7m

umlal:
umlal r0, r1, r2, r3

@ CHECK-LABEL: umlal
@ CHECK: e2 fb 03 01 umlal r0, r1, r2, r3

