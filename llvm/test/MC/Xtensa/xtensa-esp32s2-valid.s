# RUN: llvm-mc %s -triple=xtensa  -mattr=+esp32s2ops -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST:  clr_bit_gpio_out  52
# CHECK: encoding: [0x40,0x03,0x06]
clr_bit_gpio_out 52

# CHECK-INST:  get_gpio_in  a2
# CHECK: encoding: [0x20,0x30,0x06]
get_gpio_in a2

# CHECK-INST:  set_bit_gpio_out  18
# CHECK: encoding: [0x20,0x11,0x06]
set_bit_gpio_out 18

# CHECK-INST:  wr_mask_gpio_out	a3, a2
# CHECK: encoding: [0x20,0x23,0x06]
wr_mask_gpio_out	a3, a2
