# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | FileCheck %s

# Check that a branch in an end-loop packet is caught.

{ jump unknown
}:endloop0
# CHECK: 5:3: error: packet marked with `:endloop0' cannot contain instructions that modify register

{ jump unknown
}:endloop1

# CHECK: 9:3: error: packet marked with `:endloop1' cannot contain instructions that modify register
