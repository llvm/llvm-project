# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32x %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ZVE32X
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# reserved filed: vlmul[2:0]=4, vsew[2:0]=0b1xx, non-zero bits 8/9/10.
vsetvli a2, a0, 0x224
# CHECK-INST: vsetvli a2, a0, 548
# CHECK-ENCODING: [0x57,0x76,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, 0xd0
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, 0xd1
# CHECK-INST: vsetvli a2, a0, e32, m2, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x15,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, 0x50
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, mu
# CHECK-ENCODING: [0x57,0x76,0x05,0x05]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, 0x90
# CHECK-INST: vsetvli a2, a0, e32, m1, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x09]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, 144
# CHECK-INST: vsetvli a2, a0, e32, m1, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x09]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m1, ta, ma
# CHECK-INST:  vsetvli a2, a0, e32,  m1,  ta,  ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m2, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m2, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x15,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m4, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m4, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x25,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m8, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m8, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x35,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, mf2, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, mf2, ta, ma
# CHECK-ZVE32X: :[[#@LINE-2]]:17: warning: use of vtype encodings with SEW > 16 and LMUL == mf2 may not be compatible with all RVV implementations{{$}}
# CHECK-ENCODING: [0x57,0x76,0x75,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, mf4, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, mf4, ta, ma
# CHECK-ZVE32X: :[[#@LINE-2]]:17: warning: use of vtype encodings with SEW > 8 and LMUL == mf4 may not be compatible with all RVV implementations{{$}}
# CHECK-ENCODING: [0x57,0x76,0x65,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, mf8, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, mf8, ta, ma
# CHECK-ZVE32X: :[[#@LINE-2]]:22: warning: use of vtype encodings with LMUL < SEWMIN/ELEN == mf4 is reserved{{$}}
# CHECK-ENCODING: [0x57,0x76,0x55,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m1, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m1, tu, ma
# CHECK-INST: vsetvli a2, a0, e32, m1, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x09]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m1, ta, mu
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, mu
# CHECK-ENCODING: [0x57,0x76,0x05,0x05]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m1, tu, mu
# CHECK-INST: vsetvli a2, a0, e32, m1
# CHECK-ENCODING: [0x57,0x76,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, 0
# CHECK-INST: vsetvli a2, a0, e8, m1, tu, mu
# CHECK-ENCODING: [0x57,0x76,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e16
# CHECK-INST: vsetvli a2, a0, e16, m1, tu, mu
# CHECK-ENCODING: [0x57,0x76,0x85,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, m2
# CHECK-INST: vsetvli a2, a0, e8, m2, tu, mu
# CHECK-ENCODING: [0x57,0x76,0x15,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, ta
# CHECK-INST: vsetvli a2, a0, e8, m1, ta, mu
# CHECK-ENCODING: [0x57,0x76,0x05,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, ma
# CHECK-INST: vsetvli a2, a0, e8, m1, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, m2, ta, ma
# CHECK-INST: vsetvli a2, a0, e8, m2, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x15,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m2, ma
# CHECK-INST: vsetvli a2, a0, e32, m2, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x15,0x09]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m2, ta
# CHECK-INST: vsetvli a2, a0, e32, m2, ta, mu
# CHECK-ENCODING: [0x57,0x76,0x15,0x05]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, ta, ma
# CHECK-INST: vsetvli a2, a0, e8, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, m2, ma
# CHECK-INST: vsetvli a2, a0, e8, m2, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x15,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, m2, ta
# CHECK-INST: vsetvli a2, a0, e8, m2, ta, mu
# CHECK-ENCODING: [0x57,0x76,0x15,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, ma
# CHECK-INST: vsetvli a2, a0, e32, m1, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x09]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, ta
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, mu
# CHECK-ENCODING: [0x57,0x76,0x05,0x05]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvli a2, a0, e32, m2
# CHECK-INST: vsetvli a2, a0, e32, m2, tu, mu
# CHECK-ENCODING: [0x57,0x76,0x15,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetvl a2, a0, a1
# CHECK-INST: vsetvl a2, a0, a1
# CHECK-ENCODING: [0x57,0x76,0xb5,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

# reserved filed: vlmul[2:0]=4, vsew[2:0]=0b1xx, non-zero bits 8/9/10.
vsetivli a2, 0, 0x224
# CHECK-INST: vsetivli a2, 0, 548
# CHECK-ENCODING: [0x57,0x76,0x40,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 0, 0xd0
# CHECK-INST: vsetivli a2, 0, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x00,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 15, 0xd0
# CHECK-INST: vsetivli a2, 15, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x07,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 15, 208
# CHECK-INST: vsetivli a2, 15, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x07,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 0, e32, m1, ta, ma
# CHECK-INST: vsetivli a2, 0, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x00,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 15, e32, m1, ta, ma
# CHECK-INST: vsetivli a2, 15, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x07,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 31, e32, m1, ta, ma
# CHECK-INST: vsetivli a2, 31, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x0f,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, 0
# CHECK-INST: vsetivli a2, 1, e8, m1, tu, mu
# CHECK-ENCODING: [0x57,0xf6,0x00,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, e16
# CHECK-INST: vsetivli a2, 1, e16, m1, tu, mu
# CHECK-ENCODING: [0x57,0xf6,0x80,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, m2
# CHECK-INST:  vsetivli a2, 1, e8, m2, tu, mu
# CHECK-ENCODING: [0x57,0xf6,0x10,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, ta
# CHECK-INST: vsetivli a2, 1, e8, m1, ta, mu
# CHECK-ENCODING: [0x57,0xf6,0x00,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, ma
# CHECK-INST: vsetivli a2, 1, e8, m1, tu, ma
# CHECK-ENCODING: [0x57,0xf6,0x00,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, m2, ta, ma
# CHECK-INST: vsetivli a2, 1, e8, m2, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x10,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, e32, ta, ma
# CHECK-INST: vsetivli a2, 1, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x00,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, e32, m2, ma
# CHECK-INST: vsetivli a2, 1, e32, m2, tu, ma
# CHECK-ENCODING: [0x57,0xf6,0x10,0xc9]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, e32, m2, ta
# CHECK-INST: vsetivli a2, 1, e32, m2, ta, mu
# CHECK-ENCODING: [0x57,0xf6,0x10,0xc5]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, ta, ma
# CHECK-INST: vsetivli a2, 1, e8, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x00,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, m2, ma
# CHECK-INST: vsetivli a2, 1, e8, m2, tu, ma
# CHECK-ENCODING: [0x57,0xf6,0x10,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, m2, ta
# CHECK-INST: vsetivli a2, 1, e8, m2, ta, mu
# CHECK-ENCODING: [0x57,0xf6,0x10,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, e32, ma
# CHECK-INST: vsetivli a2, 1, e32, m1, tu, ma
# CHECK-ENCODING: [0x57,0xf6,0x00,0xc9]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, e32, ta
# CHECK-INST: vsetivli a2, 1, e32, m1, ta, mu
# CHECK-ENCODING: [0x57,0xf6,0x00,0xc5]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsetivli a2, 1, e32, m2
# CHECK-INST: vsetivli a2, 1, e32, m2, tu, mu
# CHECK-ENCODING: [0x57,0xf6,0x10,0xc1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
