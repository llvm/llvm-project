# RUN: llvm-mc -dwarf-version 5 %s -filetype=obj -o - | llvm-dwarfdump -debug-line - | FileCheck %s
# RUN: not llvm-mc -dwarf-version 4 %s -filetype=asm -o - 2>&1 | FileCheck %s -check-prefix=ERR
# REQUIRES: default_triple
# Darwin line table is stuck on DWARF v2.
# XFAIL: darwin
# Show that ".loc 0" works in DWARF v5, gets an error for earlier versions.
        .file 0 "root.cpp"
        .file 1 "header.h"
	.loc  0 10 0
        .byte 0
        .loc  1 20 0
        .byte 0

# CHECK:      file_names[ 0]:
# CHECK-NEXT: name: "root.cpp"
# CHECK:      file_names[ 1]:
# CHECK-NEXT: name: "header.h"
# CHECK:      Address Line Column File
# CHECK-NEXT: -------
# CHECK-NEXT: 0x{{[0-9a-f]+}} 10 0 0
# CHECK-NEXT: 0x{{[0-9a-f]+}} 20 0 1
# CHECK-NEXT: 0x{{[0-9a-f]+}} 20 0 1 {{.*}} end_sequence

# ERR:      file number less than one in '.loc' directive
# ERR-NEXT: .loc 0 10 0
# ERR-NEXT: ^
