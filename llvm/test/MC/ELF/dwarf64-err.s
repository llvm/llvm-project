# RUN: not llvm-mc -g -dwarf-version 2 -dwarf64 -triple x86_64 %s -o - 2>&1 | FileCheck --check-prefix=DWARF2 %s
# RUN: not llvm-mc -g -dwarf-version 5 -dwarf64 -triple i686 %s -o - 2>&1 | FileCheck --check-prefix=I686 %s

# DWARF2: the 64-bit DWARF format is not supported for DWARF versions prior to 3
# I686: the 64-bit DWARF format is only supported for 64-bit targets
