// RUN: llvm-mc -filetype=obj -triple thumbv7-linux-gnu %s -o %t
// RUN: llvm-objdump --triple=thumbv7-linux-gnu -d %t | FileCheck %s

//PR18303
.code 16
.global edata
b edata // CHECK: b.w
.code 32

