# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vqmaccu.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmaccu.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xb3]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))
# CHECK-UNKNOWN: 5b 24 42 b3 <unknown>

sf.vqmacc.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmacc.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xb7]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))
# CHECK-UNKNOWN: 5b 24 42 b7 <unknown>

sf.vqmaccus.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmaccus.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xbb]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))
# CHECK-UNKNOWN: 5b 24 42 bb <unknown>

sf.vqmaccsu.2x8x2 v8, v4, v20
# CHECK-INST: sf.vqmaccsu.2x8x2 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xbf]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))
# CHECK-UNKNOWN: 5b 24 42 bf <unknown>

sf.vqmaccu.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmaccu.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xf3]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))
# CHECK-UNKNOWN: 5b 24 42 f3 <unknown>

sf.vqmacc.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmacc.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xf7]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))
# CHECK-UNKNOWN: 5b 24 42 f7 <unknown>

sf.vqmaccus.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmaccus.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xfb]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))
# CHECK-UNKNOWN: 5b 24 42 fb <unknown>

sf.vqmaccsu.4x8x4 v8, v4, v20
# CHECK-INST: sf.vqmaccsu.4x8x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x24,0x42,0xff]
# CHECK-ERROR: instruction requires the following: 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))
# CHECK-UNKNOWN: 5b 24 42 ff <unknown>
