# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

### Test registers with specific names like "%sp"

        subu.l %fp, %sp, %s0
        brge.l.t %sp, %sl, 1f
        ld %s63, 0x18(,%tp)
1:
        or %got, 0, %plt
        b.l (,%lr)


# CHECK:        subu.l %s9, %s11, %s0
# CHECK-NEXT:   brge.l.t %s11, %s8, .Ltmp0
# CHECK-NEXT:   ld %s63, 24(, %s14)
# CHECK-NEXT: .Ltmp0:
# CHECK-NEXT:   or %s15, 0, %s16
# CHECK-NEXT:   b.l (, %s10)

# CHECK-INST:        subu.l %s9, %s11, %s0
# CHECK-INST-NEXT:   brge.l.t %s11, %s8, 16
# CHECK-INST-NEXT:   ld %s63, 24(, %s14)
# CHECK-INST-NEXT:   or %s15, 0, %s16
# CHECK-INST-NEXT:   b.l (, %s10)
