# RUN: llvm-mc %s -triple=mips | FileCheck %s
# FIXME: Update this test when we have a more mature implementation of Mips16 in the IAS.

.text
.set mips16
.set nomips16

# CHECK: .text
# CHECK: .set mips16
# CHECK: .set nomips16
