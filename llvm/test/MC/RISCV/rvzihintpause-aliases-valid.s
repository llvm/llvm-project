# RUN: llvm-mc %s -triple=riscv32 -mattr=+zihintpause -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zihintpause \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zihintpause -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zihintpause \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zihintpause < %s \
# RUN:     | llvm-objdump --mattr=+zihintpause -d -r -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zihintpause < %s \
# RUN:     | llvm-objdump --mattr=+zihintpause -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zihintpause < %s \
# RUN:     | llvm-objdump --mattr=+zihintpause -d -r -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zihintpause < %s \
# RUN:     | llvm-objdump --mattr=+zihintpause -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# CHECK-S-OBJ-NOALIAS: fence w, 0
# CHECK-S-OBJ: pause
pause
