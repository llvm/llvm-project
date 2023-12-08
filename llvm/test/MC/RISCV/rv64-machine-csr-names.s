# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

# These machine mode CSR register names are RV32 only, but RV64
# can encode and disassemble these registers if given their value.

######################################
# Machine Protection and Translation
######################################

# pmpcfg1
# uimm12
# CHECK-INST: csrrs t2, 929, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x3a]
# CHECK-INST-ALIAS: csrr t2, 929
csrrs t2, 0x3A1, zero

# pmpcfg3
# uimm12
# CHECK-INST: csrrs t2, 931, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x3a]
# CHECK-INST-ALIAS: csrr t2, 931
csrrs t2, 0x3A3, zero

######################################
# Machine Counter and Timers
######################################
# mcycleh
# uimm12
# CHECK-INST: csrrs t2, 2944, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2944
csrrs t2, 0xB80, zero

# minstreth
# uimm12
# CHECK-INST: csrrs t2, 2946, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2946
csrrs t2, 0xB82, zero

# mhpmcounter3h
# uimm12
# CHECK-INST: csrrs t2, 2947, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2947
csrrs t2, 0xB83, zero

# mhpmcounter4h
# uimm12
# CHECK-INST: csrrs t2, 2948, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2948
csrrs t2, 0xB84, zero

# mhpmcounter5h
# uimm12
# CHECK-INST: csrrs t2, 2949, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2949
csrrs t2, 0xB85, zero

# mhpmcounter6h
# uimm12
# CHECK-INST: csrrs t2, 2950, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2950
csrrs t2, 0xB86, zero

# mhpmcounter7h
# uimm12
# CHECK-INST: csrrs t2, 2951, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2951
csrrs t2, 0xB87, zero

# mhpmcounter8h
# uimm12
# CHECK-INST: csrrs t2, 2952, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2952
csrrs t2, 0xB88, zero

# mhpmcounter9h
# uimm12
# CHECK-INST: csrrs t2, 2953, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2953
csrrs t2, 0xB89, zero

# mhpmcounter10h
# uimm12
# CHECK-INST: csrrs t2, 2954, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2954
csrrs t2, 0xB8A, zero

# mhpmcounter11h
# uimm12
# CHECK-INST: csrrs t2, 2955, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2955
csrrs t2, 0xB8B, zero

# mhpmcounter12h
# uimm12
# CHECK-INST: csrrs t2, 2956, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2956
csrrs t2, 0xB8C, zero

# mhpmcounter13h
# uimm12
# CHECK-INST: csrrs t2, 2957, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2957
csrrs t2, 0xB8D, zero

# mhpmcounter14h
# uimm12
# CHECK-INST: csrrs t2, 2958, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2958
csrrs t2, 0xB8E, zero

# mhpmcounter15h
# uimm12
# CHECK-INST: csrrs t2, 2959, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xb8]
# CHECK-INST-ALIAS: csrr t2, 2959
csrrs t2, 0xB8F, zero

# mhpmcounter16h
# uimm12
# CHECK-INST: csrrs t2, 2960, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2960
csrrs t2, 0xB90, zero

# mhpmcounter17h
# uimm12
# CHECK-INST: csrrs t2, 2961, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2961
csrrs t2, 0xB91, zero

# mhpmcounter18h
# uimm12
# CHECK-INST: csrrs t2, 2962, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2962
csrrs t2, 0xB92, zero

# mhpmcounter19h
# uimm12
# CHECK-INST: csrrs t2, 2963, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2963
csrrs t2, 0xB93, zero

# mhpmcounter20h
# uimm12
# CHECK-INST: csrrs t2, 2964, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2964
csrrs t2, 0xB94, zero

# mhpmcounter21h
# uimm12
# CHECK-INST: csrrs t2, 2965, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2965
csrrs t2, 0xB95, zero

# mhpmcounter22h
# uimm12
# CHECK-INST: csrrs t2, 2966, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2966
csrrs t2, 0xB96, zero

# mhpmcounter23h
# uimm12
# CHECK-INST: csrrs t2, 2967, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2967
csrrs t2, 0xB97, zero

# mhpmcounter24h
# uimm12
# CHECK-INST: csrrs t2, 2968, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2968
csrrs t2, 0xB98, zero

# mhpmcounter25h
# uimm12
# CHECK-INST: csrrs t2, 2969, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2969
csrrs t2, 0xB99, zero

# mhpmcounter26h
# uimm12
# CHECK-INST: csrrs t2, 2970, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2970
csrrs t2, 0xB9A, zero

# mhpmcounter27h
# uimm12
# CHECK-INST: csrrs t2, 2971, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2971
csrrs t2, 0xB9B, zero

# mhpmcounter28h
# uimm12
# CHECK-INST: csrrs t2, 2972, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2972
csrrs t2, 0xB9C, zero

# mhpmcounter29h
# uimm12
# CHECK-INST: csrrs t2, 2973, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2973
csrrs t2, 0xB9D, zero

# mhpmcounter30h
# uimm12
# CHECK-INST: csrrs t2, 2974, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2974
csrrs t2, 0xB9E, zero

# mhpmcounter31h
# uimm12
# CHECK-INST: csrrs t2, 2975, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xb9]
# CHECK-INST-ALIAS: csrr t2, 2975
csrrs t2, 0xB9F, zero
