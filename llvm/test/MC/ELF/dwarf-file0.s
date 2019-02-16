# REQUIRES: default_triple
# RUN: llvm-mc -dwarf-version 4 %s -filetype=obj -o - | llvm-dwarfdump -debug-line - | FileCheck %s --check-prefixes=CHECK,CHECK-4
# RUN: llvm-mc -dwarf-version 4 %s -filetype=asm -o - | FileCheck %s --check-prefixes=ASM,ASM-4
# RUN: llvm-mc -dwarf-version 4 %s -filetype=asm -o - 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-mc -dwarf-version 5 %s -filetype=obj -o - | llvm-dwarfdump -debug-line - | FileCheck %s --check-prefixes=CHECK,CHECK-5
# RUN: llvm-mc -dwarf-version 5 %s -filetype=asm -o - | FileCheck %s --check-prefixes=ASM,ASM-5
# Darwin is stuck on DWARF v2.
# XFAIL: darwin
        .file 0 "/test" "root.cpp"
        .file 1 "/include" "header.h"
        .file 2 "/test" "root.cpp"
# CHECK-5:     include_directories[ 0] = "/test"
# CHECK-4-NOT: include_directories[ 0]
# CHECK:       include_directories[ 1] = "/include"
# CHECK-4:     include_directories[ 2] = "/test"
# CHECK-NOT:   include_directories
# CHECK-4-NOT: file_names[ 0]
# CHECK-5:     file_names[ 0]:
# CHECK-5-NEXT: name: "root.cpp"
# CHECK-5-NEXT: dir_index: 0
# CHECK:       file_names[ 1]:
# CHECK-NEXT:  name: "header.h"
# CHECK-NEXT:  dir_index: 1
# CHECK:       file_names[ 2]:
# CHECK-NEXT:  name: "root.cpp"
# CHECK-4-NEXT: dir_index: 2
# CHECK-5-NEXT: dir_index: 0

# ASM-NOT: .file
# ASM-5:   .file 0 "/test" "root.cpp"
# ASM:     .file 1 "/include" "header.h"
# ASM-4:   .file 2 "/test" "root.cpp"
# ASM-5:   .file 2 "root.cpp"
# ASM-NOT: .file

# WARN:      file 0 not supported prior to DWARF-5
# WARN-NEXT: .file 0
# WARN-NEXT: ^
