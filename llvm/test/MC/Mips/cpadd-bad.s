# RUN: not llvm-mc -triple=mips-unknown-linux-gnu %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=mips64-unknown-linux-gnuabin32 %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=mips64-unknown-linux-gnu %s 2>&1 | FileCheck %s

  .text
  .cpadd $32
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register
  .cpadd $foo
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected register
  .cpadd bar
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected register
  .cpadd $25 foobar
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: unexpected token, expected end of statement
