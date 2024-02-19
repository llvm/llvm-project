## Test out of range immediates which are used by lsx instructions.

# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

## uimm1
vstelm.d $vr0, $a0, 8, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 1]

vstelm.d $vr0, $a0, 8, 2
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 1]

vreplvei.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 1]

vreplvei.d $vr0, $vr1, 2
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 1]

vpickve2gr.du $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 1]

vpickve2gr.du $a0, $vr1, 2
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 1]

vpickve2gr.d $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 1]

vpickve2gr.d $a0, $vr1, 2
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 1]

vinsgr2vr.d $vr0, $a0, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 1]

vinsgr2vr.d $vr0, $a0, 2
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 1]

## uimm2
vstelm.w $vr0, $a0, 4, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

vstelm.w $vr0, $a0, 4, 4
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

vreplvei.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

vreplvei.w $vr0, $vr1, 4
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

vpickve2gr.wu $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 3]

vpickve2gr.wu $a0, $vr1, 4
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 3]

vpickve2gr.w $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 3]

vpickve2gr.w $a0, $vr1, 4
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 3]

vinsgr2vr.w $vr0, $a0, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

vinsgr2vr.w $vr0, $a0, 4
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

## uimm3
vstelm.h $vr0, $a0, 2, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vstelm.h $vr0, $a0, 2, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vreplvei.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vreplvei.h $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vpickve2gr.hu $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 7]

vpickve2gr.hu $a0, $vr1, 8
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 7]

vpickve2gr.h $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

vpickve2gr.h $a0, $vr1, 8
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

vinsgr2vr.h $vr0, $a0, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vinsgr2vr.h $vr0, $a0, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vbitrevi.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vbitrevi.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vbitseti.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vbitseti.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vbitclri.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vbitclri.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

vsrari.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

vsrari.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

vsrlri.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

vsrlri.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

vsllwil.hu.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 7]

vsllwil.hu.bu $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 7]

vsllwil.h.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

vsllwil.h.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

vrotri.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

vrotri.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

vsrai.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

vsrai.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

vsrli.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

vsrli.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

vslli.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

vslli.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

vsat.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 7]

vsat.b $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 7]

vsat.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

vsat.bu $vr0, $vr1, 8
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

## uimm4
vstelm.b $vr0, $a0, 1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vstelm.b $vr0, $a0, 1, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vreplvei.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vreplvei.b $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vpickve2gr.bu $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vpickve2gr.bu $a0, $vr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vpickve2gr.b $a0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vpickve2gr.b $a0, $vr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vinsgr2vr.b $vr0, $a0, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vinsgr2vr.b $vr0, $a0, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vbitrevi.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vbitrevi.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vbitseti.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vbitseti.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vbitclri.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vbitclri.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vssrarni.bu.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

vssrarni.bu.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

vssrlrni.bu.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

vssrlrni.bu.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

vssrarni.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrarni.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrlrni.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrlrni.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrani.bu.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrani.bu.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrlni.bu.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrlni.bu.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

vssrani.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vssrani.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vssrlni.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vssrlni.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vsrarni.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vsrarni.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vsrlrni.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vsrlrni.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vsrani.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vsrani.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vsrlni.b.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vsrlni.b.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 15]

vsrari.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

vsrari.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

vsrlri.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

vsrlri.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

vsllwil.wu.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

vsllwil.wu.hu $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

vsllwil.w.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vsllwil.w.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

vrotri.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

vrotri.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

vsrai.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

vsrai.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

vsrli.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

vsrli.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

vslli.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

vslli.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

vsat.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 15]

vsat.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 15]

vsat.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

vsat.hu $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

## uimm5
vbsrl.v $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vbsrl.v $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vbsll.v $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vbsll.v $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vslti.du $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslti.du $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslti.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslti.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslti.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslti.hu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslti.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslti.bu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.du $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.du $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.hu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vslei.bu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vfrstpi.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

vfrstpi.h $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

vfrstpi.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

vfrstpi.b $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

vbitrevi.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vbitrevi.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vbitseti.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vbitseti.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vbitclri.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vbitclri.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vssrarni.hu.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

vssrarni.hu.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

vssrlrni.hu.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

vssrlrni.hu.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

vssrarni.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrarni.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrlrni.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrlrni.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrani.hu.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrani.hu.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrlni.hu.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrlni.hu.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

vssrani.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vssrani.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vssrlni.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vssrlni.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vsrarni.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vsrarni.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vsrlrni.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vsrlrni.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vsrani.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vsrani.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vsrlni.h.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vsrlni.h.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

vsrari.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsrari.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsrlri.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsrlri.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsllwil.du.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

vsllwil.du.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

vsllwil.d.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vsllwil.d.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

vrotri.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vrotri.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsrai.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vsrai.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vsrli.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vsrli.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vslli.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vslli.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vaddi.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vaddi.bu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vaddi.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vaddi.hu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vaddi.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vaddi.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vaddi.du $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vaddi.du $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.bu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.hu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.du $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsubi.du $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.bu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.hu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.du $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmaxi.du $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.bu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.bu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.hu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.hu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.du $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vmini.du $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

vsat.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 31]

vsat.w $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 31]

vsat.wu $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

vsat.wu $vr0, $vr1, 32
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

## simm5
vslti.d $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslti.d $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslti.w $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslti.w $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslti.h $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslti.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslti.b $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslti.b $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.d $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.d $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.w $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.w $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.h $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.b $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vslei.b $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.d $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.d $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.w $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.w $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.h $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.b $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vseqi.b $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.b $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.b $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.h $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.w $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.w $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.d $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmaxi.d $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.b $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.b $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.h $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.w $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.w $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.d $vr0, $vr1, -17
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

vmini.d $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

## uimm6
vbitrevi.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vbitrevi.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vbitseti.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vbitseti.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vbitclri.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vbitclri.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vssrarni.wu.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

vssrarni.wu.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

vssrlrni.wu.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

vssrlrni.wu.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

vssrarni.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrarni.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrlrni.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrlrni.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrani.wu.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrani.wu.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrlni.wu.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrlni.wu.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

vssrani.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vssrani.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vssrlni.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vssrlni.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vsrarni.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vsrarni.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vsrlrni.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vsrlrni.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

vsrani.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vsrani.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vsrlni.w.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vsrlni.w.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 63]

vsrari.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

vsrari.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

vsrlri.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

vsrlri.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

vrotri.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

vrotri.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

vsrai.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

vsrai.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

vsrli.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

vsrli.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

vslli.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

vslli.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

vsat.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 63]

vsat.d $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 63]

vsat.du $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

vsat.du $vr0, $vr1, 64
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

## uimm7
vssrarni.du.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

vssrarni.du.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

vssrlrni.du.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

vssrlrni.du.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

vssrarni.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrarni.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrlrni.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrlrni.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrani.du.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrani.du.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrlni.du.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrlni.du.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

vssrani.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vssrani.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vssrlni.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vssrlni.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vsrarni.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vsrarni.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vsrlrni.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vsrlrni.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

vsrani.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 127]

vsrani.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 127]

vsrlni.d.q $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 127]

vsrlni.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 127]

## uimm8
vextrins.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vextrins.d $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vextrins.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vextrins.w $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vextrins.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vextrins.h $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vextrins.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vextrins.b $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vpermi.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

vpermi.w $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

vshuf4i.d $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vshuf4i.d $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vshuf4i.w $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vshuf4i.w $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vshuf4i.h $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vshuf4i.h $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vshuf4i.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vshuf4i.b $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

vbitseli.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vbitseli.b $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

vandi.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

vandi.b $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

vori.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 255]

vori.b $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 255]

vxori.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

vxori.b $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

vnori.b $vr0, $vr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

vnori.b $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

## simm8
vstelm.b $vr0, $a0, -129, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-128, 127]

vstelm.b $vr0, $a0, 128, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-128, 127]

## simm8_lsl1
vstelm.h $vr0, $a0, -258, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 2 in the range [-256, 254]

vstelm.h $vr0, $a0, 256, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 2 in the range [-256, 254]

## simm8_lsl2
vstelm.w $vr0, $a0, -516, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 4 in the range [-512, 508]

vstelm.w $vr0, $a0, 512, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 4 in the range [-512, 508]

## simm8_lsl3
vstelm.d $vr0, $a0, -1032, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 8 in the range [-1024, 1016]

vstelm.d $vr0, $a0, 1024, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 8 in the range [-1024, 1016]

## simm9_lsl3
vldrepl.d $vr0, $a0, -2056
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 8 in the range [-2048, 2040]

vldrepl.d $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 8 in the range [-2048, 2040]

## simm10_lsl2
vldrepl.w $vr0, $a0, -2052
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 4 in the range [-2048, 2044]

vldrepl.w $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 4 in the range [-2048, 2044]

## simm10
vrepli.b $vr0, -513
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

vrepli.b $vr0, 512
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

vrepli.h $vr0, -513
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

vrepli.h $vr0, 512
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

vrepli.w $vr0, -513
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

vrepli.w $vr0, 512
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

vrepli.d $vr0, -513
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

vrepli.d $vr0, 512
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

## simm11_lsl1
vldrepl.h $vr0, $a0, -2050
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 2 in the range [-2048, 2046]

vldrepl.h $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 2 in the range [-2048, 2046]

## simm12
vldrepl.b $vr0, $a0, -2049
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-2048, 2047]

vldrepl.b $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-2048, 2047]

vst $vr0, $a0, -2049
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-2048, 2047]

vst $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-2048, 2047]

vld $vr0, $a0, -2049
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-2048, 2047]

vld $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-2048, 2047]

## simm13
vldi $vr0, -4097
# CHECK: :[[#@LINE-1]]:12: error: immediate must be an integer in the range [-4096, 4095]

vldi $vr0, 4096
# CHECK: :[[#@LINE-1]]:12: error: immediate must be an integer in the range [-4096, 4095]
