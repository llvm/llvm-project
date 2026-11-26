# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvkned %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvkned %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvkned --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vaesdf.vv v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesdf.vv v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x90,0xa2]

vaesdf.vs v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesdf.vs v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x90,0xa6]

vaesef.vv v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesef.vv v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x91,0xa2]
                       
vaesef.vs v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesef.vs v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x91,0xa6]

vaesdm.vv v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesdm.vv v10, v9
# CHECK-ENCODING: [0x77,0x25,0x90,0xa2]
                       
vaesdm.vs v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesdm.vs v10, v9
# CHECK-ENCODING: [0x77,0x25,0x90,0xa6]

vaesem.vv v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesem.vv v10, v9
# CHECK-ENCODING: [0x77,0x25,0x91,0xa2]
                       
vaesem.vs v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesem.vs v10, v9
# CHECK-ENCODING: [0x77,0x25,0x91,0xa6]

vaeskf1.vi v10, v9, 1 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaeskf1.vi v10, v9, 1
# CHECK-ENCODING: [0x77,0xa5,0x90,0x8a]

vaeskf1.vi v10, v9, 31 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaeskf1.vi v10, v9, 31
# CHECK-ENCODING: [0x77,0xa5,0x9f,0x8a]

vaeskf1.vi v0, v9, 1 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaeskf1.vi v0, v9, 1
# CHECK-ENCODING: [0x77,0xa0,0x90,0x8a]

vaeskf2.vi v10, v9, 2 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaeskf2.vi v10, v9, 2
# CHECK-ENCODING: [0x77,0x25,0x91,0xaa]

vaeskf2.vi v10, v9, 31 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaeskf2.vi v10, v9, 31
# CHECK-ENCODING: [0x77,0xa5,0x9f,0xaa]

vaeskf2.vi v0, v9, 2 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaeskf2.vi v0, v9, 2
# CHECK-ENCODING: [0x77,0x20,0x91,0xaa]

vaesz.vs v10, v9 # expected-error {{instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round))}}
# CHECK-INST: vaesz.vs v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x93,0xa6]
