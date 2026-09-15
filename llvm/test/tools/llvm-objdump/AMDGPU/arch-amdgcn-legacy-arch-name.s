# The architecture name was renamed from "amdgcn" to "amdgpu". Check that the
# legacy "amdgcn" name still works when passed via --arch-name.

# RUN: llvm-mc -triple=amdgcn -mcpu=gfx900 -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d --arch-name=amdgcn --mcpu=gfx900 %t.o | FileCheck %s

# CHECK: s_nop 0
s_nop 0
