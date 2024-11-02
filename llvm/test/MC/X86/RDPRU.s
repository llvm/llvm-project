/// Encoding and disassembly of rdpru.

// RUN: llvm-mc -triple i686-- --show-encoding %s |\
// RUN:   FileCheck %s --check-prefixes=CHECK,ENCODING

// RUN: llvm-mc -triple i686-- -filetype=obj %s |\
// RUN:   llvm-objdump -d - | FileCheck %s

// RUN: llvm-mc -triple x86_64-- --show-encoding %s |\
// RUN:   FileCheck %s --check-prefixes=CHECK,ENCODING

// RUN: llvm-mc -triple x86_64-- -filetype=obj %s |\
// RUN:   llvm-objdump -d - | FileCheck %s

// CHECK: rdpru
// ENCODING:  encoding: [0x0f,0x01,0xfd]
rdpru
