# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvkned %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvkned %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvkned --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvkned %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vaesdf.vv v10, v9
# CHECK-INST: vaesdf.vv v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x90,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a290a577 <unknown>

vaesdf.vs v10, v9
# CHECK-INST: vaesdf.vs v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x90,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a690a577 <unknown>

vaesef.vv v10, v9
# CHECK-INST: vaesef.vv v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x91,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a291a577 <unknown>
                       
vaesef.vs v10, v9
# CHECK-INST: vaesef.vs v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x91,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a691a577 <unknown>

vaesdm.vv v10, v9
# CHECK-INST: vaesdm.vv v10, v9
# CHECK-ENCODING: [0x77,0x25,0x90,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a2902577 <unknown>
                       
vaesdm.vs v10, v9
# CHECK-INST: vaesdm.vs v10, v9
# CHECK-ENCODING: [0x77,0x25,0x90,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a6902577 <unknown>

vaesem.vv v10, v9
# CHECK-INST: vaesem.vv v10, v9
# CHECK-ENCODING: [0x77,0x25,0x91,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a2912577 <unknown>
                       
vaesem.vs v10, v9
# CHECK-INST: vaesem.vs v10, v9
# CHECK-ENCODING: [0x77,0x25,0x91,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a6912577 <unknown>

vaeskf1.vi v10, v9, 1
# CHECK-INST: vaeskf1.vi v10, v9, 1
# CHECK-ENCODING: [0x77,0xa5,0x90,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: 8a90a577 <unknown>

vaeskf1.vi v10, v9, 31
# CHECK-INST: vaeskf1.vi v10, v9, 31
# CHECK-ENCODING: [0x77,0xa5,0x9f,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: 8a9fa577 <unknown>

vaeskf2.vi v10, v9, 2
# CHECK-INST: vaeskf2.vi v10, v9, 2
# CHECK-ENCODING: [0x77,0x25,0x91,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: aa912577 <unknown>

vaeskf2.vi v10, v9, 31
# CHECK-INST: vaeskf2.vi v10, v9, 31
# CHECK-ENCODING: [0x77,0xa5,0x9f,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: aa9fa577 <unknown>

vaesz.vs v10, v9
# CHECK-INST: vaesz.vs v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x93,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN: a693a577 <unknown>
