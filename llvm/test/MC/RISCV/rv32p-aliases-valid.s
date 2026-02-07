# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-p -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-p \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-p < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r -M no-aliases --mattr=+experimental-p - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-p < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r --mattr=+experimental-p - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# Tests for aliases that are available on both RV32 and RV64

# CHECK-S-OBJ-NOALIAS: padd.bs a0, zero, a1
# CHECK-S-OBJ: pmv.bs a0, a1
pmv.bs a0, a1

# CHECK-S-OBJ-NOALIAS: padd.hs a2, zero, a3
# CHECK-S-OBJ: pmv.hs a2, a3
pmv.hs a2, a3

# CHECK-S-OBJ-NOALIAS: psub.b a0, zero, a1
# CHECK-S-OBJ: pneg.b a0, a1
pneg.b a0, a1

# CHECK-S-OBJ-NOALIAS: psub.h a2, zero, a3
# CHECK-S-OBJ: pneg.h a2, a3
pneg.h a2, a3

# CHECK-S-OBJ-NOALIAS: pabd.b a0, a1, zero
# CHECK-S-OBJ: pabs.b a0, a1
pabs.b a0, a1

# CHECK-S-OBJ-NOALIAS: pabd.h a2, a3, zero
# CHECK-S-OBJ: pabs.h a2, a3
pabs.h a2, a3

# CHECK-S-OBJ-NOALIAS: ppaire.b a0, a1, zero
# CHECK-S-OBJ: pzext.h.b a0, a1
pzext.h.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmseq.b a0, a1, zero
# CHECK-S-OBJ: pmseqz.b a0, a1
pmseqz.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmseq.h a2, a3, zero
# CHECK-S-OBJ: pmseqz.h a2, a3
pmseqz.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmsltu.b a0, zero, a1
# CHECK-S-OBJ: pmsnez.b a0, a1
pmsnez.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmsltu.h a2, zero, a3
# CHECK-S-OBJ: pmsnez.h a2, a3
pmsnez.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmslt.b a0, a1, zero
# CHECK-S-OBJ: pmsltz.b a0, a1
pmsltz.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmslt.h a2, a3, zero
# CHECK-S-OBJ: pmsltz.h a2, a3
pmsltz.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmslt.b a0, zero, a1
# CHECK-S-OBJ: pmsgtz.b a0, a1
pmsgtz.b a0, a1

# CHECK-S-OBJ-NOALIAS: pmslt.h a2, zero, a3
# CHECK-S-OBJ: pmsgtz.h a2, a3
pmsgtz.h a2, a3

# CHECK-S-OBJ-NOALIAS: pmslt.b a0, a2, a1
# CHECK-S-OBJ: pmslt.b a0, a2, a1
pmsgt.b a0, a1, a2

# CHECK-S-OBJ-NOALIAS: pmslt.h a3, a5, a4
# CHECK-S-OBJ: pmslt.h a3, a5, a4
pmsgt.h a3, a4, a5

# CHECK-S-OBJ-NOALIAS: pmsltu.b a0, a2, a1
# CHECK-S-OBJ: pmsltu.b a0, a2, a1
pmsgtu.b a0, a1, a2

# CHECK-S-OBJ-NOALIAS: pmsltu.h a3, a5, a4
# CHECK-S-OBJ: pmsltu.h a3, a5, a4
pmsgtu.h a3, a4, a5

# Tests for RV32-only aliases (pair operations)

# CHECK-S-OBJ-NOALIAS: padd.dbs a0, zero, a1
# CHECK-S-OBJ: pmv.dbs a0, a1
pmv.dbs a0, a1

# CHECK-S-OBJ-NOALIAS: padd.dhs a2, zero, a3
# CHECK-S-OBJ: pmv.dhs a2, a3
pmv.dhs a2, a3

# CHECK-S-OBJ-NOALIAS: padd.dws a4, zero, a5
# CHECK-S-OBJ: pmv.dws a4, a5
pmv.dws a4, a5

# CHECK-S-OBJ-NOALIAS: pabd.db a0, a2, zero
# CHECK-S-OBJ: pabs.db a0, a2
pabs.db a0, a2

# CHECK-S-OBJ-NOALIAS: pabd.dh a4, a6, zero
# CHECK-S-OBJ: pabs.dh a4, a6
pabs.dh a4, a6

# CHECK-S-OBJ-NOALIAS: psub.db a0, zero, a2
# CHECK-S-OBJ: pneg.db a0, a2
pneg.db a0, a2

# CHECK-S-OBJ-NOALIAS: psub.dh a4, zero, a6
# CHECK-S-OBJ: pneg.dh a4, a6
pneg.dh a4, a6

# CHECK-S-OBJ-NOALIAS: psub.dw a0, zero, a2
# CHECK-S-OBJ: pneg.dw a0, a2
pneg.dw a0, a2

# CHECK-S-OBJ-NOALIAS: ppaire.db a0, a2, zero
# CHECK-S-OBJ: pzext.dh.b a0, a2
pzext.dh.b a0, a2

# CHECK-S-OBJ-NOALIAS: ppaire.dh a4, a6, zero
# CHECK-S-OBJ: pzext.dw.h a4, a6
pzext.dw.h a4, a6

# CHECK-S-OBJ-NOALIAS: pmseq.db a0, a2, zero
# CHECK-S-OBJ: pmseqz.db a0, a2
pmseqz.db a0, a2

# CHECK-S-OBJ-NOALIAS: pmseq.dh a4, a6, zero
# CHECK-S-OBJ: pmseqz.dh a4, a6
pmseqz.dh a4, a6

# CHECK-S-OBJ-NOALIAS: pmseq.dw a0, a2, zero
# CHECK-S-OBJ: pmseqz.dw a0, a2
pmseqz.dw a0, a2

# CHECK-S-OBJ-NOALIAS: pmsltu.db a0, zero, a2
# CHECK-S-OBJ: pmsnez.db a0, a2
pmsnez.db a0, a2

# CHECK-S-OBJ-NOALIAS: pmsltu.dh a4, zero, a6
# CHECK-S-OBJ: pmsnez.dh a4, a6
pmsnez.dh a4, a6

# CHECK-S-OBJ-NOALIAS: pmsltu.dw a0, zero, a2
# CHECK-S-OBJ: pmsnez.dw a0, a2
pmsnez.dw a0, a2

# CHECK-S-OBJ-NOALIAS: pmslt.db a0, a2, zero
# CHECK-S-OBJ: pmsltz.db a0, a2
pmsltz.db a0, a2

# CHECK-S-OBJ-NOALIAS: pmslt.dh a4, a6, zero
# CHECK-S-OBJ: pmsltz.dh a4, a6
pmsltz.dh a4, a6

# CHECK-S-OBJ-NOALIAS: pmslt.dw a0, a2, zero
# CHECK-S-OBJ: pmsltz.dw a0, a2
pmsltz.dw a0, a2

# CHECK-S-OBJ-NOALIAS: pmslt.db a0, zero, a2
# CHECK-S-OBJ: pmsgtz.db a0, a2
pmsgtz.db a0, a2

# CHECK-S-OBJ-NOALIAS: pmslt.dh a4, zero, a6
# CHECK-S-OBJ: pmsgtz.dh a4, a6
pmsgtz.dh a4, a6

# CHECK-S-OBJ-NOALIAS: pmslt.dw a0, zero, a2
# CHECK-S-OBJ: pmsgtz.dw a0, a2
pmsgtz.dw a0, a2

# CHECK-S-OBJ-NOALIAS: pmslt.db a0, a4, a2
# CHECK-S-OBJ: pmslt.db a0, a4, a2
pmsgt.db a0, a2, a4

# CHECK-S-OBJ-NOALIAS: pmslt.dh a6, a2, a0
# CHECK-S-OBJ: pmslt.dh a6, a2, a0
pmsgt.dh a6, a0, a2

# CHECK-S-OBJ-NOALIAS: pmslt.dw a4, a6, a2
# CHECK-S-OBJ: pmslt.dw a4, a6, a2
pmsgt.dw a4, a2, a6

# CHECK-S-OBJ-NOALIAS: pmsltu.db a0, a4, a2
# CHECK-S-OBJ: pmsltu.db a0, a4, a2
pmsgtu.db a0, a2, a4

# CHECK-S-OBJ-NOALIAS: pmsltu.dh a6, a2, a0
# CHECK-S-OBJ: pmsltu.dh a6, a2, a0
pmsgtu.dh a6, a0, a2

# CHECK-S-OBJ-NOALIAS: pmsltu.dw a4, a6, a2
# CHECK-S-OBJ: pmsltu.dw a4, a6, a2
pmsgtu.dw a4, a2, a6

# Tests for RV32-only scalar aliases (MSEQ/MSLT/MSLTU and NEGD)

# CHECK-S-OBJ-NOALIAS: subd a4, zero, a6
# CHECK-S-OBJ: negd a4, a6
negd a4, a6

# CHECK-S-OBJ-NOALIAS: mseq a0, a1, zero
# CHECK-S-OBJ: mseqz a0, a1
mseqz a0, a1

# CHECK-S-OBJ-NOALIAS: msltu a0, zero, a1
# CHECK-S-OBJ: msnez a0, a1
msnez a0, a1

# CHECK-S-OBJ-NOALIAS: mslt a0, a1, zero
# CHECK-S-OBJ: msltz a0, a1
msltz a0, a1

# CHECK-S-OBJ-NOALIAS: mslt a0, zero, a1
# CHECK-S-OBJ: msgtz a0, a1
msgtz a0, a1

# CHECK-S-OBJ-NOALIAS: mslt a0, a2, a1
# CHECK-S-OBJ: mslt a0, a2, a1
msgt a0, a1, a2

# CHECK-S-OBJ-NOALIAS: msltu a3, a5, a4
# CHECK-S-OBJ: msltu a3, a5, a4
msgtu a3, a4, a5
