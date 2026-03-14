# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o - | llvm-readobj -S - | FileCheck %s --strict-whitespace

        .zero 4
foo:
        .zero 4
        .org foo+16

.bss
        .zero 1
# .org is a zero initializer and can appear in a SHT_NOBITS section.
        .org .bss+5

# CHECK:      Section {
# CHECK:        Name: .text
# CHECK:        Size:
# CHECK-SAME:         {{ 20$}}

# CHECK:      Section {
# CHECK:        Name: .bss
# CHECK:        Size:
# CHECK-SAME:         {{ 5$}}
