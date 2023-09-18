## Test out of range immediates which are used by lasx instructions.

# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

## uimm1
xvrepl128vei.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 1]

xvrepl128vei.d $xr0, $xr1, 2
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 1]

## uimm2
xvpickve.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

xvpickve.d $xr0, $xr1, 4
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

xvinsve0.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

xvinsve0.d $xr0, $xr1, 4
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 3]

xvinsgr2vr.d $xr0, $a0, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 3]

xvinsgr2vr.d $xr0, $a0, 4
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 3]

xvpickve2gr.d $a0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 3]

xvpickve2gr.d $a0, $xr1, 4
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 3]

xvpickve2gr.du $a0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 3]

xvpickve2gr.du $a0, $xr1, 4
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 3]

xvstelm.d $xr0, $a0, 8, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 3]

xvstelm.d $xr0, $a0, 8, 4
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 3]

xvrepl128vei.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 3]

xvrepl128vei.w $xr0, $xr1, 4
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 3]

## uimm3
xvpickve.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

xvpickve.w $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

xvinsve0.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

xvinsve0.w $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 7]

xvinsgr2vr.w $xr0, $a0, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvinsgr2vr.w $xr0, $a0, 8
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvpickve2gr.wu $a0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 7]

xvpickve2gr.wu $a0, $xr1, 8
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 7]

xvpickve2gr.w $a0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 7]

xvpickve2gr.w $a0, $xr1, 8
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 7]

xvstelm.w $xr0, $a0, 4, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvstelm.w $xr0, $a0, 4, 8
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvrepl128vei.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 7]

xvrepl128vei.h $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 7]

xvbitrevi.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvbitrevi.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvbitseti.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvbitseti.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvbitclri.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvbitclri.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 7]

xvsrari.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 7]

xvsrari.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 7]

xvsrlri.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 7]

xvsrlri.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 7]

xvsllwil.hu.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 7]

xvsllwil.hu.bu $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 7]

xvsllwil.h.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 7]

xvsllwil.h.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 7]

xvrotri.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 7]

xvrotri.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 7]

xvsrai.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

xvsrai.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

xvsrli.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

xvsrli.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

xvslli.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

xvslli.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

xvsat.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

xvsat.b $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 7]

xvsat.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

xvsat.bu $xr0, $xr1, 8
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 7]

## uimm4
xvstelm.h $xr0, $a0, 2, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvstelm.h $xr0, $a0, 2, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvrepl128vei.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvrepl128vei.b $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvbitrevi.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvbitrevi.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvbitseti.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvbitseti.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvbitclri.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvbitclri.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvssrarni.bu.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvssrarni.bu.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvssrlrni.bu.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvssrlrni.bu.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvssrarni.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrarni.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrlrni.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrlrni.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrani.bu.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrani.bu.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrlni.bu.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrlni.bu.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 15]

xvssrani.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvssrani.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvssrlni.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvssrlni.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvsrarni.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvsrarni.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvsrlrni.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvsrlrni.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvsrani.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvsrani.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvsrlni.b.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvsrlni.b.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 15]

xvsrari.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 15]

xvsrari.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 15]

xvsrlri.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 15]

xvsrlri.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 15]

xvsllwil.wu.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvsllwil.wu.hu $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 15]

xvsllwil.w.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvsllwil.w.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 15]

xvrotri.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 15]

xvrotri.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 15]

xvsrai.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

xvsrai.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

xvsrli.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

xvsrli.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

xvslli.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

xvslli.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

xvsat.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

xvsat.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

xvsat.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

xvsat.hu $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 15]

## uimm5
xvstelm.b $xr0, $a0, 1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvstelm.b $xr0, $a0, 1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvbsrl.v $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvbsrl.v $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvbsll.v $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvbsll.v $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvslti.du $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslti.du $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslti.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslti.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslti.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslti.hu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslti.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslti.bu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.du $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.du $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.hu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvslei.bu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvfrstpi.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

xvfrstpi.h $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

xvfrstpi.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

xvfrstpi.b $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

xvbitrevi.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvbitrevi.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvbitseti.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvbitseti.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvbitclri.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvbitclri.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvssrarni.hu.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 31]

xvssrarni.hu.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 31]

xvssrlrni.hu.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 31]

xvssrlrni.hu.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 31]

xvssrarni.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrarni.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrlrni.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrlrni.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrani.hu.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrani.hu.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrlni.hu.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrlni.hu.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 31]

xvssrani.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvssrani.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvssrlni.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvssrlni.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvsrarni.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvsrarni.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvsrlrni.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvsrlrni.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvsrani.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvsrani.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvsrlni.h.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvsrlni.h.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]

xvsrari.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsrari.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsrlri.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsrlri.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsllwil.du.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 31]

xvsllwil.du.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 31]

xvsllwil.d.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvsllwil.d.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 31]

xvrotri.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvrotri.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsrai.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvsrai.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvsrli.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvsrli.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvslli.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvslli.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvaddi.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvaddi.bu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvaddi.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvaddi.hu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvaddi.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvaddi.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvaddi.du $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvaddi.du $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.bu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.hu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.du $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsubi.du $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.bu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.hu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.du $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmaxi.du $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.bu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.bu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.hu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.hu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.du $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvmini.du $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 31]

xvsat.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

xvsat.w $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

xvsat.wu $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

xvsat.wu $xr0, $xr1, 32
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]

## simm5
xvslti.d $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslti.d $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslti.w $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslti.w $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslti.h $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslti.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslti.b $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslti.b $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.d $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.d $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.w $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.w $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.h $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.b $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvslei.b $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.d $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.d $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.w $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.w $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.h $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.b $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvseqi.b $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.b $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.b $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.h $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.w $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.w $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.d $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmaxi.d $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.b $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.b $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.h $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.w $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.w $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.d $xr0, $xr1, -17
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

xvmini.d $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

## uimm6
xvbitrevi.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvbitrevi.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvbitseti.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvbitseti.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvbitclri.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvbitclri.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvssrarni.wu.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 63]

xvssrarni.wu.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 63]

xvssrlrni.wu.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 63]

xvssrlrni.wu.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 63]

xvssrarni.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrarni.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrlrni.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrlrni.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrani.wu.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrani.wu.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrlni.wu.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrlni.wu.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 63]

xvssrani.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvssrani.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvssrlni.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvssrlni.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvsrarni.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvsrarni.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvsrlrni.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvsrlrni.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 63]

xvsrani.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvsrani.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvsrlni.w.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvsrlni.w.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]

xvsrari.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 63]

xvsrari.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 63]

xvsrlri.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 63]

xvsrlri.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 63]

xvrotri.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 63]

xvrotri.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 63]

xvsrai.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

xvsrai.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

xvsrli.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

xvsrli.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

xvslli.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

xvslli.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

xvsat.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

xvsat.d $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 63]

xvsat.du $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

xvsat.du $xr0, $xr1, 64
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

## uimm7
xvssrarni.du.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 127]

xvssrarni.du.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 127]

xvssrlrni.du.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 127]

xvssrlrni.du.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 127]

xvssrarni.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrarni.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrlrni.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrlrni.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrani.du.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrani.du.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrlni.du.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrlni.du.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 127]

xvssrani.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvssrani.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvssrlni.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvssrlni.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvsrarni.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvsrarni.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvsrlrni.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvsrlrni.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:26: error: immediate must be an integer in the range [0, 127]

xvsrani.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

xvsrani.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

xvsrlni.d.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

xvsrlni.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

## uimm8
xvextrins.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvextrins.d $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvextrins.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvextrins.w $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvextrins.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvextrins.h $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvextrins.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvextrins.b $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvpermi.q $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

xvpermi.q $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

xvpermi.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

xvpermi.d $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

xvpermi.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

xvpermi.w $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

xvshuf4i.d $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvshuf4i.d $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvshuf4i.w $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvshuf4i.w $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvshuf4i.h $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvshuf4i.h $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvshuf4i.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvshuf4i.b $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 255]

xvbitseli.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvbitseli.b $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 255]

xvandi.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

xvandi.b $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

xvori.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

xvori.b $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 255]

xvxori.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

xvxori.b $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

xvnori.b $xr0, $xr1, -1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

xvnori.b $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

## simm8
xvstelm.b $xr0, $a0, -129, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-128, 127]

xvstelm.b $xr0, $a0, 128, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-128, 127]

## simm8_lsl1
xvstelm.h $xr0, $a0, -258, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 2 in the range [-256, 254]

xvstelm.h $xr0, $a0, 256, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 2 in the range [-256, 254]

## simm8_lsl2
xvstelm.w $xr0, $a0, -516, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 4 in the range [-512, 508]

xvstelm.w $xr0, $a0, 512, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 4 in the range [-512, 508]

## simm8_lsl3
xvstelm.d $xr0, $a0, -1032, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 8 in the range [-1024, 1016]

xvstelm.d $xr0, $a0, 1024, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 8 in the range [-1024, 1016]

## simm9_lsl3
xvldrepl.d $xr0, $a0, -2056
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 8 in the range [-2048, 2040]

xvldrepl.d $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 8 in the range [-2048, 2040]

## simm10_lsl2
xvldrepl.w $xr0, $a0, -2052
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 4 in the range [-2048, 2044]

xvldrepl.w $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 4 in the range [-2048, 2044]

## simm10
xvrepli.b $xr0, -513
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

xvrepli.b $xr0, 512
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

xvrepli.h $xr0, -513
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

xvrepli.h $xr0, 512
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

xvrepli.w $xr0, -513
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

xvrepli.w $xr0, 512
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

xvrepli.d $xr0, -513
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

xvrepli.d $xr0, 512
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

## simm11_lsl1
xvldrepl.h $xr0, $a0, -2050
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 2 in the range [-2048, 2046]

xvldrepl.h $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 2 in the range [-2048, 2046]

## simm12
xvldrepl.b $xr0, $a0, -2049
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [-2048, 2047]

xvldrepl.b $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [-2048, 2047]

xvst $xr0, $a0, -2049
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-2048, 2047]

xvst $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-2048, 2047]

xvld $xr0, $a0, -2049
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-2048, 2047]

xvld $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-2048, 2047]

## simm13
xvldi $xr0, -4097
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [-4096, 4095]

xvldi $xr0, 4096
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [-4096, 4095]
