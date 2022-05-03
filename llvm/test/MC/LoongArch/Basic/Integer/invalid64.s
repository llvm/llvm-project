## Test invalid instructions on loongarch64 target.

# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

## Out of range immediates
## uimm2_plus1
alsl.wu $a0, $a0, $a0, 0
# CHECK: :[[#@LINE-1]]:24: error: immediate must be an integer in the range [1, 4]
alsl.d $a0, $a0, $a0, 5
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [1, 4]

## uimm3
bytepick.d $a0, $a0, $a0, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 7]
bytepick.d $a0, $a0, $a0, 8
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 7]

## uimm6
slli.d $a0, $a0, -1
# CHECK: :[[#@LINE-1]]:18: error: immediate must be an integer in the range [0, 63]
srli.d $a0, $a0, -1
# CHECK: :[[#@LINE-1]]:18: error: immediate must be an integer in the range [0, 63]
srai.d $a0, $a0, 64
# CHECK: :[[#@LINE-1]]:18: error: immediate must be an integer in the range [0, 63]
rotri.d $a0, $a0, 64
# CHECK: :[[#@LINE-1]]:19: error: immediate must be an integer in the range [0, 63]
bstrins.d $a0, $a0, 63, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 63]
bstrpick.d $a0, $a0, 64, 0
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 63]

## simm12
addi.d $a0, $a0, -2049
# CHECK: :[[#@LINE-1]]:18: error: immediate must be an integer in the range [-2048, 2047]
lu52i.d $a0, $a0, -2049
# CHECK: :[[#@LINE-1]]:19: error: immediate must be an integer in the range [-2048, 2047]
ld.wu $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-2048, 2047]
st.d $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-2048, 2047]

## simm14_lsl2
ldptr.w $a0, $a0, -32772
# CHECK: :[[#@LINE-1]]:19: error: immediate must be a multiple of 4 in the range [-32768, 32764]
ldptr.d $a0, $a0, -32772
# CHECK: :[[#@LINE-1]]:19: error: immediate must be a multiple of 4 in the range [-32768, 32764]
stptr.w $a0, $a0, -32769
# CHECK: :[[#@LINE-1]]:19: error: immediate must be a multiple of 4 in the range [-32768, 32764]
stptr.d $a0, $a0, -32769
# CHECK: :[[#@LINE-1]]:19: error: immediate must be a multiple of 4 in the range [-32768, 32764]
ll.w $a0, $a0, 32767
# CHECK: :[[#@LINE-1]]:16: error: immediate must be a multiple of 4 in the range [-32768, 32764]
sc.w $a0, $a0, 32768
# CHECK: :[[#@LINE-1]]:16: error: immediate must be a multiple of 4 in the range [-32768, 32764]

## simm16
addu16i.d $a0, $a0, -32769
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-32768, 32767]
addu16i.d $a0, $a0, 32768
# CHECK: :[[#@LINE-1]]:21: error: immediate must be an integer in the range [-32768, 32767]

## simm20
lu32i.d $a0, -0x80001
# CHECK: :[[#@LINE-1]]:14: error: immediate must be an integer in the range [-524288, 524287]
pcaddu18i $a0, 0x80000
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-524288, 524287]

## msbd < lsbd
# CHECK: :[[#@LINE+1]]:21: error: msb is less than lsb
bstrins.d $a0, $a0, 1, 2
# CHECK:            ^~~~

# CHECK: :[[#@LINE+1]]:22: error: msb is less than lsb
bstrpick.d $a0, $a0, 32, 63
# CHECK:             ^~~~~~
