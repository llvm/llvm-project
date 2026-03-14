# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s | llvm-readelf -s - | FileCheck %s

# CHECK:    Value         Size Type   Bind   Vis     Ndx Name
# CHECK: 0000000000000004    4 OBJECT GLOBAL DEFAULT COM C
        .comm   C,4,4
        .comm   C,4,4
