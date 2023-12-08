// RUN: llvm-mc -triple aarch64-elf -filetype=obj %s -o - | llvm-objdump -d -r - | FileCheck %s

.macro do_add sz
        add     v0.\sz, v0.\sz, v0.\sz
.endm

do_add 8h
// CHECK:  add     v0.8h, v0.8h, v0.8h
