# RUN: llvm-mc -filetype=obj -arch=hexagon -mv65 -mattr=+hvxv65,+hvx-length128b %s | llvm-objdump -d --mattr=+hvx - | FileCheck %s

# CHECK: vhist
vhist
