// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o %t
// RUN: llvm-objdump -d %t | FileCheck %s

.global foo	
pushw $foo // CHECK: pushw
