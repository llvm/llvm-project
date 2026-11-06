# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

sf.vqmaccu.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmaccu.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xb3]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))}}

sf.vqmacc.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmacc.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xb7]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))}}

sf.vqmaccus.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmaccus.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xbb]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))}}

sf.vqmaccsu.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmaccsu.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xbf]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))}}

sf.vqmaccu.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmaccu.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xf3]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))}}

sf.vqmacc.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmacc.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xf7]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))}}

sf.vqmaccus.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmaccus.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xfb]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))}}

sf.vqmaccsu.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmaccsu.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xff]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))}}

sf.vqmaccu.4x8x4 v0, v4, v20
# CHECK-INST: sf.vqmaccu.4x8x4 v0, v4, v20
# CHECK-ENCODING: [0x5b,0x20,0x42,0xf3]
# expected-error@-3 {{instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))}}
