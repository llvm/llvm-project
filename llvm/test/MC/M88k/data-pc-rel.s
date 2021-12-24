# RUN: llvm-mc -triple m88k-unknown-openbsdk -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -triple m88k-unknown-openbsd -show-encoding %s -o - \
# RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

# RELOC: R_88K_DISP26 label 0x0
# INSTR: br label
# FIXUP: fixup A - offset: 0, value: label, kind: FK_88K_DISP26
br label

# RELOC: R_88K_DISP26 label 0x0
# INSTR: bsr label
# FIXUP: fixup A - offset: 0, value: label, kind: FK_88K_DISP26
bsr label

# RELOC: R_88K_DISP16 label 0x0
# INSTR: bb0 0, %r1, label
# FIXUP: fixup A - offset: 0, value: label, kind: FK_88K_DISP16
bb0 0, %r1, label

# RELOC: R_88K_DISP16 label 0x0
# INSTR: bb1 0, %r1, label
# FIXUP: fixup A - offset: 0, value: label, kind: FK_88K_DISP16
bb1 0, %r1, label

# RELOC: R_88K_DISP16 label 0x0
# INSTR: bcnd eq0, %r1, label
# FIXUP: fixup A - offset: 0, value: label, kind: FK_88K_DISP16
bcnd eq0, %r1, label
