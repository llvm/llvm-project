# RUN: llvm-mc -dwarf-version 4 %s -filetype=obj -o - | llvm-dwarfdump -debug-line - | FileCheck %s --check-prefixes=CHECK,CHECK-4
# RUN: llvm-mc -dwarf-version 5 %s -filetype=obj -o - | llvm-dwarfdump -debug-line - | FileCheck %s --check-prefixes=CHECK,CHECK-5
# REQUIRES: default_triple
# Darwin is stuck on DWARF v2.
# XFAIL: darwin
        .file 0 "root.cpp"
        .file 1 "header.h"
        .file 2 "root.cpp"
# CHECK-5:     include_directories[ 0] = ""
# CHECK-4-NOT: include_directories
# CHECK-4-NOT: file_names[ 0]
# CHECK-5:     file_names[ 0]:
# CHECK-5-NEXT: name: "root.cpp"
# CHECK-5-NEXT: dir_index: 0
# CHECK:       file_names[ 1]:
# CHECK-NEXT:  name: "header.h"
# CHECK-NEXT:  dir_index: 0
# CHECK:       file_names[ 2]:
# CHECK-NEXT:  name: "root.cpp"
# CHECK-NEXT:  dir_index: 0
