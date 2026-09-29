# REQUIRES: x86-registered-target

.type foo,@function
.size foo,12
foo:
    .space 10
    nop
    nop

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o -g

# RUN: llvm-symbolizer 0xa 0xb --print-address --obj=%t.o \
# RUN:   | FileCheck %s --check-prefix=NORMAL
# RUN: llvm-symbolizer 0x10a 0x10b --print-address --adjust-vma 0x100 --obj=%t.o \
# RUN:   | FileCheck %s --check-prefix=ADJUST

# Show that we can handle addresses less than the offset.
# RUN: llvm-symbolizer 0xa 0xb --print-address --adjust-vma 0xc --obj=%t.o \
# RUN:   | FileCheck %s --check-prefix=OVERFLOW

# NORMAL:      0xa
# NORMAL-NEXT: foo
# NORMAL-NEXT: adjust-vma.s:7:0
# NORMAL-EMPTY:
# NORMAL-NEXT: 0xb
# NORMAL-NEXT: foo
# NORMAL-NEXT: adjust-vma.s:8:0

# ADJUST:      0x10a
# ADJUST-NEXT: foo
# ADJUST-NEXT: adjust-vma.s:7:0
# ADJUST-EMPTY:
# ADJUST-NEXT: 0x10b
# ADJUST-NEXT: foo
# ADJUST-NEXT: adjust-vma.s:8:0

# OVERFLOW:      0xa
# OVERFLOW-NEXT: ??
# OVERFLOW-NEXT: ??
