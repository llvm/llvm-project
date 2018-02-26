# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
# RUN: obj2yaml %t.o | FileCheck %s

# CHECK: Relocations:
# CHECK:   Symbol:          .text
# CHECK: Symbols:
# CHECK:   - Name:            .text

        .quad .text
