// RUN: llvm-mc -triple aarch64-windows-gnu -filetype obj -o %t.obj %s
// RUN: llvm-objdump -d %t.obj | FileCheck %s
// RUN: llvm-mc -triple aarch64-windows-msvc -filetype obj -o %t.obj %s
// RUN: llvm-objdump -d %t.obj | FileCheck %s

func:
// Check that the nop instruction after the semicolon also is handled
nop; nop
add x0, x0, #42

// CHECK:  0:       d503201f     nop
// CHECK:  4:       d503201f     nop
// CHECK:  8:       9100a800     add x0, x0, #42
