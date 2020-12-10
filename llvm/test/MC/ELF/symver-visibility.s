# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# CHECK:      NOTYPE GLOBAL HIDDEN    {{[1-9]}} def@@v1
# CHECK-NEXT: NOTYPE GLOBAL PROTECTED UND       undef@v1

.protected undef
.symver undef, undef@@@v1
call undef

.globl def
.hidden def
.symver def, def@@@v1
def:
