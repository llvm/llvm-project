# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readelf -h -s - | FileCheck %s --check-prefix=OBJ

# ASM: .type unique,@gnu_unique_object

# OBJ: OS/ABI: UNIX - GNU
# OBJ: Type   Bind   Vis     Ndx Name
# OBJ: OBJECT UNIQUE DEFAULT [[#]] unique

.data
.globl unique
.type unique, @gnu_unique_object
unique:
