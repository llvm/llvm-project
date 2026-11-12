# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -h -s %t | FileCheck %s --check-prefix=OBJ
# RUN: llvm-objdump --no-print-imm-hex -d -r --no-show-raw-insn %t | FileCheck %s --check-prefix=DISASM

# ASM: .type unique,@gnu_unique_object

# OBJ: OS/ABI: UNIX - GNU
# OBJ: Type   Bind   Vis     Ndx Name
# OBJ: OBJECT UNIQUE DEFAULT [[#]] unique

# DISASM-LABEL: <.text>:
# DISASM-NEXT:    movl $1, 0
## unique has a non-local binding. Reference unique instead of .data
# DISASM-NEXT:      R_X86_64_32S unique

  movl $1, unique

.data
.globl unique
.type unique, @gnu_unique_object
unique:
