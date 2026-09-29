# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
# RUN: rm -f %t
# RUN: llvm-ar rcs %t %t.o
# RUN: llvm-nm --print-armap %t | FileCheck %s

# Test that weak undefined symbols don't show up in the archive symbol
# table.

.global foo
foo:
.weak bar
.quad bar

# CHECK: Archive map
# CHECK-NEXT: foo in archive-symbol-table.s.tmp.o
# CHECK-NOT: in
# CHECK: archive-symbol-table.s.tmp.o
# CHECK-NEXT: w bar
# CHECK-NEXT: T foo
