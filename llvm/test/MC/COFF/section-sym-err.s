# RUN: not llvm-mc -filetype=obj -triple x86_64-windows-gnu %s -o %t.o 2>&1 | FileCheck %s --implicit-check-not=error:

.section foo0,"xr",discard,foo1
foo0:
foo1:

# CHECK: error: symbol 'foo0' is already defined
