# RUN: llvm-mc -filetype=obj -triple mips--gnu -g %s \
# RUN:   | llvm-objdump -s -section=.eh_frame - | FileCheck --check-prefix=O32 %s
# RUN: llvm-mc -filetype=obj -triple mips64--gnuabin32 -g %s \
# RUN:   | llvm-objdump -s -section=.eh_frame - | FileCheck --check-prefix=N32 %s
# RUN: llvm-mc -filetype=obj -triple mips64--gnuabi64 -g %s \
# RUN:   | llvm-objdump -s -section=.eh_frame - | FileCheck --check-prefix=N64 %s

# O32: 0000 00000010 00000000 017a5200 017c1f01
# O32: 0010 0b0d1d00 00000010 00000018 00000000
# O32: 0020 00000004 00000000

# N32: 0000 00000010 00000000 017a5200 017c1f01
# N32: 0010 0b0d1d00 00000010 00000018 00000000
# N32: 0020 00000004 00000000

# N64: 0000 00000010 00000000 017a5200 01781f01
# N64: 0010 0c0d1d00 00000018 00000018 00000000
# N64: 0020 00000000 00000000 00000004 00000000

foo:
  .cfi_startproc
  nop
  .cfi_endproc
