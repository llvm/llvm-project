## Test out of range immediates which are used by lsx instructions.

# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

## uimm1
vreplvei.d $vr0, $vr1, 2
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 1]

## uimm4
vsat.h $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 15]

## simm5
vseqi.b $vr0, $vr1, 16
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-16, 15]

## uimm7
vsrlni.d.q $vr0, $vr1, 128
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [0, 127]

## simm8
vpermi.w $vr0, $vr1, 256
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 255]

## simm8_lsl1
vstelm.h $vr0, $a0, 255, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 2 in the range [-256, 254]

## simm8_lsl2
vstelm.w $vr0, $a0, 512, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 4 in the range [-512, 508]

## simm10
vrepli.b $vr0, 512
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-512, 511]

## simm8_lsl3
vstelm.d $vr0, $a0, 1024, 1
# CHECK: :[[#@LINE-1]]:21: error: immediate must be a multiple of 8 in the range [-1024, 1016]

## simm9_lsl3
vldrepl.d $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 8 in the range [-2048, 2040]

## simm10_lsl2
vldrepl.w $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 4 in the range [-2048, 2044]

## simm11_lsl1
vldrepl.h $vr0, $a0, 2048
# CHECK: :[[#@LINE-1]]:22: error: immediate must be a multiple of 2 in the range [-2048, 2046]

## simm13
vldi $vr0, 4096
# CHECK: :[[#@LINE-1]]:12: error: immediate must be an integer in the range [-4096, 4095]
