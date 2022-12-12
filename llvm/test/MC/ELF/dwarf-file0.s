# XFAIL: target={{.*}}-aix{{.*}}
# UNSUPPORTED: -zos
# REQUIRES: object-emission
# RUN: llvm-mc -dwarf-version 4 %s -filetype=obj -o - | llvm-dwarfdump -debug-line - | FileCheck %s
# RUN: llvm-mc -dwarf-version 4 %s --fatal-warnings -o - | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -dwarf-version 5 %s -filetype=obj -o - | llvm-dwarfdump -debug-line - | FileCheck %s
# RUN: llvm-mc -dwarf-version 5 %s -o - | FileCheck %s --check-prefix=ASM

## If the DWARF version is less than 5, .file 0 upgrades the version to 5.
        .file 0 "/test" "root.cpp"
        .file 1 "/include" "header.h"
        .file 2 "/test" "root.cpp"
# CHECK:       include_directories[  0] = "/test"
# CHECK-NEXT:  include_directories[  1] = "/include"
# CHECK:       file_names[  0]:
# CHECK-NEXT:             name: "root.cpp"
# CHECK-NEXT:        dir_index: 0
# CHECK-NEXT:  file_names[  1]:
# CHECK-NEXT:             name: "header.h"
# CHECK-NEXT:        dir_index: 1

# ASM:     .file 0 "/test" "root.cpp"
# ASM:     .file 1 "/include" "header.h"
# ASM-NOT: .file
