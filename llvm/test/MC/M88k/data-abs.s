# RUN: llvm-mc -triple m88k-unknown-openbsdk -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -triple m88k-unknown-openbsd -show-encoding %s -o - \
# RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

# TODO The following instruction are also absolute memory reference
#      Needs assembler support for %hi16 and %lo16
#        or.u     %r9,%r0,%hi16(mem)
#        ld       %r9,%r9,%lo16(mem)

# RELOC: R_88K_DISP26 - 0xA
# INSTR: br 10
# FIXUP: fixup A - offset: 0, value: 10, kind: FK_88K_DISP26
br 10

# RELOC: R_88K_DISP26 - 0x14
# INSTR: bsr 20
# FIXUP: fixup A - offset: 0, value: 20, kind: FK_88K_DISP26
bsr 20

# RELOC: R_88K_DISP16 - 0x1E
# INSTR: bb0 0, %r1, 30
# FIXUP: fixup A - offset: 0, value: 30, kind: FK_88K_DISP16
bb0 0, %r1, 30

# RELOC: R_88K_DISP16 - 0x28
# INSTR: bb1 0, %r1, 40
# FIXUP: fixup A - offset: 0, value: 40, kind: FK_88K_DISP16
bb1 0, %r1, 40

# RELOC: R_88K_DISP16 - 0x32
# INSTR: bcnd eq0, %r1, 50
# FIXUP: fixup A - offset: 0, value: 50, kind: FK_88K_DISP16
bcnd eq0, %r1, 50
