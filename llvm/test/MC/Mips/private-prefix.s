# RUN: llvm-mc %s -triple=mips   --target-abi=o32 | FileCheck %s --check-prefix=O32
# RUN: llvm-mc %s -triple=mips   --target-abi=n32 | FileCheck %s --check-prefix=N32
# RUN: llvm-mc %s -triple=mips   --target-abi=n64 | FileCheck %s --check-prefix=N64
# RUN: llvm-mc %s -triple=mips64 --target-abi=o32 | FileCheck %s --check-prefix=O32
# RUN: llvm-mc %s -triple=mips64 --target-abi=n32 | FileCheck %s --check-prefix=N32
# RUN: llvm-mc %s -triple=mips64 --target-abi=n64 | FileCheck %s --check-prefix=N64
# RUN: llvm-mc %s -triple=mips             | FileCheck %s --check-prefix=O32
# RUN: llvm-mc %s -triple=mips-gnu         | FileCheck %s --check-prefix=O32
# RUN: llvm-mc %s -triple=mips-gnuabin32   | FileCheck %s --check-prefix=N32
# RUN: llvm-mc %s -triple=mips-gnuabi64    | FileCheck %s --check-prefix=O32
# RUN: llvm-mc %s -triple=mips64           | FileCheck %s --check-prefix=N64
# RUN: llvm-mc %s -triple=mips64-gnu       | FileCheck %s --check-prefix=N64
# RUN: llvm-mc %s -triple=mips64-gnuabin32 | FileCheck %s --check-prefix=N32
# RUN: llvm-mc %s -triple=mips64-gnuabi64  | FileCheck %s --check-prefix=N64

# Checks if correct private global and label prefixes are used based on target
# options.

# O32: $tmp0:
# N32: .Ltmp0:
# N64: .Ltmp0:
li.d  $4, 1.12345
