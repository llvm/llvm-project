## Test out of range immediates which are used by lasx instructions.

# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

## uimm1
xvrepl128vei.d $xr0, $xr1, 2
# CHECK: :[[#@LINE-1]]:28: error: immediate must be an integer in the range [0, 1]

## uimm4
xvsat.h $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [0, 15]

## simm5
xvseqi.b $xr0, $xr1, 16
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [-16, 15]

## uimm7
xvsrlni.d.q $xr0, $xr1, 128
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 127]

## simm8
xvpermi.w $xr0, $xr1, 256
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [0, 255]

## simm8_lsl1
xvstelm.h $xr0, $a0, 255, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 2 in the range [-256, 254]

## simm8_lsl2
xvstelm.w $xr0, $a0, 512, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 4 in the range [-512, 508]

## simm10
xvrepli.b $xr0, 512
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-512, 511]

## simm8_lsl3
xvstelm.d $xr0, $a0, 1024, 1
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 8 in the range [-1024, 1016]

## simm9_lsl3
xvldrepl.d $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 8 in the range [-2048, 2040]

## simm10_lsl2
xvldrepl.w $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 4 in the range [-2048, 2044]

## simm11_lsl1
xvldrepl.h $xr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:23: error: immediate must be a multiple of 2 in the range [-2048, 2046]

## simm13
xvldi $xr0, 4096
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [-4096, 4095]
