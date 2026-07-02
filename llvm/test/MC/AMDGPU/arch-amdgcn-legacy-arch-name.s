# The architecture name was renamed from "amdgcn" to "amdgpu". Check that the
# legacy "amdgcn" name still works when passed via -arch.

# RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s

# CHECK: s_nop 0 ; encoding: [0x00,0x00,0x80,0xbf]
s_nop 0
