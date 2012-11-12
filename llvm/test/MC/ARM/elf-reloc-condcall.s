// RUN: llvm-mc -triple=armv7-linux-gnueabi -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        bleq some_label
        bl some_label
        blx some_label
        beq some_label
        b some_label
// OBJ: .rel.text

// OBJ: 'r_offset', 0x00000000
// OBJ-NEXT:  'r_sym', 0x000004
// OBJ-NEXT: 'r_type', 0x1d

// OBJ: 'r_offset', 0x00000004
// OBJ-NEXT:  'r_sym', 0x000004
// OBJ-NEXT: 'r_type', 0x1c

// OBJ: 'r_offset', 0x00000008
// OBJ-NEXT:  'r_sym', 0x000004
// OBJ-NEXT: 'r_type', 0x1c

// OBJ: 'r_offset', 0x0000000c
// OBJ-NEXT:  'r_sym', 0x000004
// OBJ-NEXT: 'r_type', 0x1d

// OBJ: 'r_offset', 0x00000010
// OBJ-NEXT:  'r_sym', 0x000004
// OBJ-NEXT: 'r_type', 0x1d

// OBJ: .symtab
// OBJ: Symbol 4
// OBJ-NEXT: some_label
