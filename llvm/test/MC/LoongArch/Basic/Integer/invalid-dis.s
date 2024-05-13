# Test that disassembler rejects data smaller than 4 bytes.

# RUN: llvm-mc --filetype=obj --triple=loongarch32 < %s \
# RUN:     | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 < %s \
# RUN:     | llvm-objdump -d - | FileCheck %s

# CHECK: 11 <unknown>
# CHECK: 22 <unknown>
.2byte 0x2211
