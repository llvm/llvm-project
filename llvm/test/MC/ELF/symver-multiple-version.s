# RUN: not llvm-mc -filetype=obj -triple x86_64 %s -o %t 2>&1 | FileCheck %s

# CHECK:      error: multiple symbol versions defined for defined1
# CHECK-NEXT: error: multiple symbol versions defined for defined2
# CHECK-NEXT: error: multiple symbol versions defined for defined3
# CHECK-NEXT: error: multiple symbol versions defined for undef

defined1:
defined2:
defined3:

.symver defined1, defined1@1
.symver defined1, defined1@2
.symver defined2, defined2@1
.symver defined2, defined2@2
.symver defined3, defined@@@1
.symver defined3, defined@@@2

.symver undef, undef@1
.symver undef, undef@2
