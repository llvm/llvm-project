# RUN: llvm-mc %s --triple=loongarch32 --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch32 --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s

# ASM-AND-OBJ: fcmp.caf.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x10,0x0c]
fcmp.caf.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cun.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x14,0x0c]
fcmp.cun.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.ceq.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x12,0x0c]
fcmp.ceq.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cueq.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x16,0x0c]
fcmp.cueq.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.clt.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x11,0x0c]
fcmp.clt.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cult.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x15,0x0c]
fcmp.cult.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cle.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x13,0x0c]
fcmp.cle.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cule.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x17,0x0c]
fcmp.cule.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cne.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x18,0x0c]
fcmp.cne.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cor.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x1a,0x0c]
fcmp.cor.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cune.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x1c,0x0c]
fcmp.cune.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.saf.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x10,0x0c]
fcmp.saf.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sun.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x14,0x0c]
fcmp.sun.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.seq.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x12,0x0c]
fcmp.seq.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sueq.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x16,0x0c]
fcmp.sueq.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.slt.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x11,0x0c]
fcmp.slt.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sult.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x15,0x0c]
fcmp.sult.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sle.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x13,0x0c]
fcmp.sle.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sule.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x17,0x0c]
fcmp.sule.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sne.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x18,0x0c]
fcmp.sne.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sor.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x1a,0x0c]
fcmp.sor.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sune.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x1c,0x0c]
fcmp.sune.s $fcc0, $fa0, $fa1
