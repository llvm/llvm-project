# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=ret %s | llvm-readobj -S - | FileCheck %s

## Check we only pad in a text section

# CHECK-LABEL:  Name: text
# CHECK:        AddressAlignment: 32
.section text, "ax"
ret

# CHECK-LABEL:  Name: excluded
# CHECK:        AddressAlignment: 1
.section excluded, "e"
ret

# CHECK-LABEL:  Name: tls
# CHECK:        AddressAlignment: 1
.section tls, "awT"
ret
