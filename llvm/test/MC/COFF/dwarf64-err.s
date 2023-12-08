# RUN: not llvm-mc -dwarf64 -triple x86_64-unknown-windows-gnu %s -o - 2>&1 | FileCheck %s

# CHECK: the 64-bit DWARF format is not supported for x86_64-unknown-windows-gnu
