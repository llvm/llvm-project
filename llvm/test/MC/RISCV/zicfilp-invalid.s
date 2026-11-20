# RUN: llvm-mc --verify -triple riscv32 -mattr=+experimental-zicfilp -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+experimental-zicfilp -M no-aliases -show-encoding < %s
#
# lpad is always available even without Zicfilp (riscv-non-isa/riscv-elf-psabi-doc#474).
#
# RUN: llvm-mc --verify -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -M no-aliases -show-encoding < %s

lpad 1048576 # expected-error {{immediate must be an integer in the range [0, 1048575]}}
