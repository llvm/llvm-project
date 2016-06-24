# The effects of .ent on the .pdr section are tested in mips-pdr*.s. Test
# everything else here.
#
# RUN: llvm-mc -mcpu=mips32 -triple mips-unknown-unknown %s | \
# RUN:     FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -mcpu=mips32 -triple mips-unknown-unknown %s | \
# RUN:     llvm-readobj -symbols | \
# RUN:     FileCheck -check-prefixes=OBJ,OBJ-32 %s
#
# RUN: llvm-mc -mcpu=mips32 -mattr=micromips -triple mips-unknown-unknown %s | \
# RUN:     FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -mcpu=mips32 -mattr=micromips \
# RUN:     -triple mips-unknown-unknown %s | \
# RUN:     llvm-readobj -symbols | \
# RUN:     FileCheck -check-prefixes=OBJ,OBJ-MM %s
#
    .ent a
a:

# ASM: .ent a
# ASM: a:

# OBJ:     Name: a
# OBJ:     Value: 0x0
# OBJ:     Size: 0
# OBJ:     Binding: Local
# OBJ:     Type: Function
# OBJ:     Other: 0
# OBJ:     Section: .text
# OBJ: }

    .ent b
b:
    nop
    nop
    .end b

# ASM: .ent b
# ASM: b:

# OBJ:     Name: b
# OBJ:     Value: 0x0
# OBJ-32:  Size: 8
# FIXME: microMIPS uses the 4-byte nop instead of the 2-byte nop.
# OBJ-MM:  Size: 8
# OBJ:     Binding: Local
# OBJ:     Type: Function
# OBJ:     Other: 0
# OBJ:     Section: .text
# OBJ: }
