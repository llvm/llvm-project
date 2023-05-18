# RUN: llvm-mc -filetype=obj -triple mips--gnu -g %s \
# RUN:   | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=O32 %s
# RUN: llvm-mc -filetype=obj -triple mips64--gnuabin32 -g %s \
# RUN:   | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=N32 %s
# RUN: llvm-mc -filetype=obj -triple mips64--gnuabi64 -g %s \
# RUN:   | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=N64 %s

# O32: addr_size = 0x04
# N32: addr_size = 0x04
# N64: addr_size = 0x08

foo:
  nop
