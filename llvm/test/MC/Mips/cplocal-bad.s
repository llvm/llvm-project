# RUN: not llvm-mc -triple=mips-unknown-linux-gnu %s 2>&1 \
# RUN:   | FileCheck -check-prefix=O32 %s
# RUN: not llvm-mc -triple=mips64-unknown-linux-gnuabin32 %s 2>&1 \
# RUN:   | FileCheck -check-prefix=NABI %s
# RUN: not llvm-mc -triple=mips64-unknown-linux-gnu %s 2>&1 \
# RUN:   | FileCheck -check-prefix=NABI %s

  .text
  .cplocal $32
# O32:  :[[@LINE-1]]:{{[0-9]+}}: error: .cplocal is allowed only in N32 or N64 mode
# NABI: :[[@LINE-2]]:{{[0-9]+}}: error: invalid register
  .cplocal $foo
# O32:  :[[@LINE-1]]:{{[0-9]+}}: error: .cplocal is allowed only in N32 or N64 mode
# NABI: :[[@LINE-2]]:{{[0-9]+}}: error: expected register containing global pointer
  .cplocal bar
# O32:  :[[@LINE-1]]:{{[0-9]+}}: error: .cplocal is allowed only in N32 or N64 mode
# NABI: :[[@LINE-2]]:{{[0-9]+}}: error: expected register containing global pointer
  .cplocal $25 foobar
# O32:  :[[@LINE-1]]:{{[0-9]+}}: error: .cplocal is allowed only in N32 or N64 mode
# NABI: :[[@LINE-2]]:{{[0-9]+}}: error: unexpected token, expected end of statement
