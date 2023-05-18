// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

x: .indirect_symbol _y
// CHECK-ERROR: 4:4: error: indirect symbol not in a symbol pointer or stub section
