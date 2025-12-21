# RUN: llvm-mc %s -triple=riscv64 -mattr=+ztso -M no-aliases 2>&1 | FileCheck %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+ztso -M no-aliases 2>&1 | FileCheck %s

# Note: Ztso doesn't add or remove any instructions, so this is basically
# just checking that a) we accepted the attribute name, and b) codegen did
# not change.  The ELF header flag is tested in elf-flags.s

# CHECK-NOT: not a recognized feature

# CHECK: fence iorw, iorw
fence iorw, iorw
# CHECK: fence.tso
fence.tso
# CHECK: fence.i
fence.i
