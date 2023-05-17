# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s

## Support for the 'D' extension implies support for 'F'
# ASM-AND-OBJ: fcmp.caf.s $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x10,0x0c]
fcmp.caf.s $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.caf.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x20,0x0c]
fcmp.caf.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cun.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x24,0x0c]
fcmp.cun.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.ceq.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x22,0x0c]
fcmp.ceq.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cueq.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x26,0x0c]
fcmp.cueq.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.clt.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x21,0x0c]
fcmp.clt.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cult.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x25,0x0c]
fcmp.cult.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cle.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x23,0x0c]
fcmp.cle.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cule.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x27,0x0c]
fcmp.cule.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cne.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x28,0x0c]
fcmp.cne.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cor.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x2a,0x0c]
fcmp.cor.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.cune.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x04,0x2c,0x0c]
fcmp.cune.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.saf.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x20,0x0c]
fcmp.saf.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sun.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x24,0x0c]
fcmp.sun.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.seq.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x22,0x0c]
fcmp.seq.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sueq.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x26,0x0c]
fcmp.sueq.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.slt.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x21,0x0c]
fcmp.slt.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sult.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x25,0x0c]
fcmp.sult.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sle.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x23,0x0c]
fcmp.sle.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sule.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x27,0x0c]
fcmp.sule.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sne.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x28,0x0c]
fcmp.sne.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sor.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x2a,0x0c]
fcmp.sor.d $fcc0, $fa0, $fa1

# ASM-AND-OBJ: fcmp.sune.d $fcc0, $fa0, $fa1
# ASM: encoding: [0x00,0x84,0x2c,0x0c]
fcmp.sune.d $fcc0, $fa0, $fa1
