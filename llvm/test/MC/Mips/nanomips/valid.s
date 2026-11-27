# Instructions that are valid
#
# RUN: llvm-mc %s -triple=nanomips-elf -show-encoding -show-inst | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple nanomips-elf %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex --triple nanomips-elf -dr - \
# RUN:   | FileCheck --check-prefixes DISAS %s

    .text
    # CHECK: .text
    .set noat
    # reg3-reg3 arithmetic, 16-bit
    addu    $a1, $s2, $a3   # CHECK: addu   $a1, $s2, $a3   # encoding: [0xaa,0xb3]
                            # CHECK-NEXT: # <MCInst #{{.*}} ADDu16_NM
                            # DISAS: {{.*}}: aa b3  addu $a1, $s2, $a3
    subu    $a0, $a2, $a3   # CHECK: subu   $a0, $a2, $a3   # encoding: [0xe9,0xb3]
                            # CHECK-NEXT: # <MCInst #{{.*}} SUBu16_NM
                            # DISAS: {{.*}}: e9 b3  subu $a0, $a2, $a3

    # reg4-reg4 arithmetic, 16-bit
    addu    $a1, $a1, $a7   # CHECK: addu $a1, $a1, $a7 # encoding: [0xa3,0x3c]
                            # CHECK-NEXT: # <MCInst #{{.*}} ADDu4x4_NM
                            # DISAS: {{.*}}: a3 3c  addu $a1, $a1, $a7
    mul $s2, $s2, $a3       # CHECK: mul $s2, $s2, $a3  # encoding: [0x4f,0x3e]
                            # CHECK-NEXT: # <MCInst #{{.*}} MUL4x4_NM
                            # DISAS: {{.*}}: 4f 3e  mul $s2, $s2, $a3

    # reg-reg arithmetic, 32-bit
    add $a1, $a2, $a3       # CHECK: add $a1, $a2, $a3  # encoding: [0xe6,0x20,0x10,0x29]
                            # CHECK-NEXT: # <MCInst #{{.*}} ADD_NM
                            # DISAS: {{.*}}: e6 20 10 29    add $a1, $a2, $a3
    addu    $a1, $s2, $t3   # CHECK: addu $a1, $s2, $t3 # encoding: [0xf2,0x21,0x50,0x29]
                            # CHECK-NEXT: # <MCInst #{{.*}} ADDu_NM
                            # DISAS: {{.*}}: f2 21 50 29    addu    $a1, $s2, $t3
    sub $a1, $s2, $a3       # CHECK: sub $a1, $s2, $a3  # encoding: [0xf2,0x20,0x90,0x29]
                            # CHECK-NEXT: # <MCInst #{{.*}} SUB_NM
                            # DISAS: {{.*}}: f2 20 90 29    sub $a1, $s2, $a3
    subu    $a0, $a2, $t3   # CHECK: subu $a0, $a2, $t3 # encoding: [0xe6,0x21,0xd0,0x21]
                            # CHECK-NEXT: # <MCInst #{{.*}} SUBu_NM
                            # DISAS: {{.*}}: e6 21 d0 21    subu    $a0, $a2, $t
    mul $t1, $s2, $a3       # CHECK: mul $t1, $s2, $a3  # encoding: [0xf2,0x20,0x18,0x68]
                            # CHECK-NEXT: # <MCInst #{{.*}} MUL_NM
                            # DISAS: {{.*}}: f2 20 18 68    mul $t1, $s2, $a3
    muh $a1, $s2, $a3       # CHECK: muh $a1, $s2, $a3  # encoding: [0xf2,0x20,0x58,0x28]
                            # CHECK-NEXT: # <MCInst #{{.*}} MUH_NM
                            # DISAS: {{.*}}: f2 20 58 28    muh $a1, $s2, $a3
    mulu    $k1, $s2, $a3   # CHECK: mulu $k1, $s2, $a3 # encoding: [0xf2,0x20,0x98,0xd8]
                            # CHECK-NEXT: # <MCInst #{{.*}} MULU_NM
                            # DISAS: {{.*}} f2 20 98 d8     mulu    $k1, $s2, $a3
    muhu    $a1, $s2, $a3   # CHECK: muhu $a1, $s2, $a3 # encoding: [0xf2,0x20,0xd8,0x28]
                            # CHECK-NEXT: # <MCInst #{{.*}} MUHU_NM
                            # DISAS: {{.*}} f2 20 d8 28     muhu    $a1, $s2, $a3
    div $a1, $s2, $t0       # CHECK: div $a1, $s2, $t0  # encoding: [0x92,0x21,0x18,0x29]
                            # CHECK-NEXT: # <MCInst #{{.*}} DIV_NM
                            # DISAS: {{.*}} 92 21 18 29     div $a1, $s2, $t0
    divu    $t4, $s2, $a3   # CHECK: divu $t4, $s2, $a3 # encoding: [0xf2,0x20,0x98,0x11]
                            # CHECK-NEXT: # <MCInst #{{.*}} DIVU_NM
                            # DISAS: {{.*}} f2 20 98 11     divu    $t4, $s2, $a3

    mod $a1, $t2, $a3       # CHECK: mod $a1, $t2, $a3  # encoding: [0xee,0x20,0x58,0x29]
                            # CHECK-NEXT: # <MCInst #{{.*}} MOD_NM
                            # DISAS: {{.*}} ee 20 58 29     mod $a1, $t2, $a3

    modu    $a1, $s2, $t5   # CHECK: modu $a1, $s2, $t5 # encoding: [0x72,0x20,0xd8,0x29]
                            # CHECK-NEXT: # <MCInst #{{.*}} MODU_NM
                            # DISAS: {{.*}} 72 20 d8 29     modu    $a1, $s2, $t5

    # reg-reg logic, 16-bit
    and $a1, $a3, $a1       # CHECK: and $a1, $a3, $a1  # encoding: [0xf8,0x52]
                            # CHECK-NEXT: # <MCInst #{{.*}} AND16_NM
                            # DISAS: {{.*}}  f8 52          and $a1, $a3, $a1
    or  $s2, $a3, $s2       # CHECK: or $s2, $a3, $s2   # encoding: [0x7c,0x51]
                            # CHECK-NEXT: # <MCInst #{{.*}} OR16_NM
                            # DISAS: {{.*}}  7c 51          or  $s2, $a3, $s2
    xor $a3, $s3, $a3       # CHECK: xor $a3, $s3, $a3  # encoding: [0xb4,0x53]
                            # CHECK-NEXT: # <MCInst #{{.*}} XOR16_NM
                            # DISAS: {{.*}}  b4 53          xor $a3, $s3, $a3
    not $a3, $s3            # CHECK: not $a3, $s3       # encoding: [0xb0,0x53]
                            # CHECK-NEXT: # <MCInst #{{.*}} NOT16_NM
                            # DISAS: {{.*}}  b0 53          not $a3, $s3

    # reg-reg logic, 32-bit
    and $a1, $s4, $a3       # CHECK: and $a1, $s4, $a3  # encoding: [0xf4,0x20,0x50,0x2a]
                            # CHECK-NEXT: # <MCInst #{{.*}} AND_NM
                            # DISAS: {{.*}}  f4 20 50 2a    and $a1, $s4, $a3
    or  $a1, $t2, $a3       # CHECK: or $a1, $t2, $a3   # encoding: [0xee,0x20,0x90,0x2a]
                            # CHECK-NEXT: # <MCInst #{{.*}} OR_NM
                            # DISAS: {{.*}}  ee 20 90 2a    or  $a1, $t2, $a3
    nor $a3, $t3, $s3       # CHECK: nor $a3, $t3, $s3  # encoding: [0x6f,0x22,0xd0,0x3a]
                            # CHECK-NEXT: # <MCInst #{{.*}} NOR_NM
                            # DISAS: {{.*}}  6f 22 d0 3a    nor $a3, $t3, $s3
    xor $t0, $s2, $a3       # CHECK: xor $t0, $s2, $a3  # encoding: [0xf2,0x20,0x10,0x63]
                            # CHECK-NEXT: # <MCInst #{{.*}} XOR_NM
                            # DISAS: {{.*}}  f2 20 10 63    xor $t0, $s2, $a3

    # reg-reg bitwise, 32-bit
    sllv    $a1, $a7, $a3   # CHECK: sllv $a1, $a7, $a3 # encoding: [0xeb,0x20,0x10,0x28]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLLV_NM
                            # DISAS: {{.*}}  eb 20 10 28    sllv    $a1, $a7, $a3
    srlv    $a1, $s1, $a3   # CHECK: srlv $a1, $s1, $a3 # encoding: [0xf1,0x20,0x50,0x28]
                            # CHECK-NEXT: # <MCInst #{{.*}} SRLV_NM
                            # DISAS: {{.*}}  f1 20 50 28    srlv    $a1, $s1, $a3
    srav    $a1, $s2, $a3   # CHECK: srav $a1, $s2, $a3 # encoding: [0xf2,0x20,0x90,0x28]
                            # CHECK-NEXT: # <MCInst #{{.*}} SRAV_NM
                            # DISAS: {{.*}}  f2 20 90 28    srav    $a1, $s2, $a3
    rotrv   $a1, $s2, $a3   # CHECK: rotrv $a1, $s2, $a3    # encoding: [0xf2,0x20,0xd0,0x28]
                            # CHECK-NEXT: # <MCInst #{{.*}} ROTRV_NM
                            # DISAS: {{.*}}  f2 20 d0 28    rotrv   $a1, $s2, $a3
    slt $a1, $s2, $a3       # CHECK: slt $a1, $s2, $a3  # encoding: [0xf2,0x20,0x50,0x2b]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLT_NM
                            # DISAS: {{.*}}  f2 20 50 2b    slt $a1, $s2, $a3
    sltu    $a1, $s2, $a3   # CHECK: sltu $a1, $s2, $a3 # encoding: [0xf2,0x20,0x90,0x2b]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLTU_NM
                            # DISAS: {{.*}}  f2 20 90 2b    sltu    $a1, $s2, $a3
    sov $a1, $s3, $t2       # CHECK: sov $a1, $s3, $t2  # encoding: [0xd3,0x21,0xd0,0x2b]
                            # CHECK-NEXT: # <MCInst #{{.*}} SOV_NM
                            # DISAS: {{.*}}  d3 21 d0 2b    sov $a1, $s3, $t2

    # reg-imm logic, 16-bit
    sll $a1, $s2, 1         # CHECK: sll $a1, $s2, 1    # encoding: [0xa1,0x32]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLL16_NM
                            # DISAS: {{.*}}  a1 32          sll $a1, $s2, 1{{$}}
    srl $s1, $a3, 8         # CHECK: srl $s1, $a3, 8    # encoding: [0xf8,0x30]
                            # CHECK-NEXT: # <MCInst #{{.*}} SRL16_NM
                            # DISAS: {{.*}}  f8 30          srl $s1, $a3, 8{{$}}
    andi    $a0, $a1, 0     # CHECK: andi $a0, $a1, 0x0 # encoding: [0x50,0xf2]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI16_NM
                            # DISAS: {{.*}}  50 f2          andi    $a0, $a1, 0x0{{$}}
    andi    $a1, $a2, 1     # CHECK: andi $a1, $a2, 0x1 # encoding: [0xe1,0xf2]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI16_NM
                            # DISAS: {{.*}}  e1 f2          andi    $a1, $a2, 0x1{{$}}
    andi    $a2, $a3, 6     # CHECK: andi $a2, $a3, 0x6 # encoding: [0x76,0xf3]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI16_NM
                            # DISAS: {{.*}}  76 f3          andi    $a2, $a3, 0x6{{$}}
    andi    $s3, $a3, 0xe   # CHECK: andi $s3, $a3, 0xe # encoding: [0xfe,0xf1]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI16_NM
                            # DISAS: {{.*}}  fe f1          andi    $s3, $a3, 0xe{{$}}

    # reg-imm logic, 32-bit
    andi    $a4, $a1, 0     # CHECK: andi $a4, $a1, 0x0 # encoding: [0x05,0x81,0x00,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  05 81 00 20    andi    $a4, $a1, 0x0{{$}}
    andi    $a1, $t2, 1     # CHECK: andi $a1, $t2, 0x1 # encoding: [0xae,0x80,0x01,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  ae 80 01 20    andi    $a1, $t2, 0x1{{$}}
    andi    $a2, $t3, 6     # CHECK: andi $a2, $t3, 0x6 # encoding: [0xcf,0x80,0x06,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  cf 80 06 20    andi    $a2, $t3, 0x6{{$}}
    andi    $a6, $s0, 0xff  # CHECK: andi $a6, $s0, 0xff    # encoding: [0x50,0x81,0xff,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  50 81 ff 20    andi    $a6, $s0, 0xff{{$}}
    andi    $s3, $a4, 0xe   # CHECK: andi $s3, $a4, 0xe # encoding: [0x68,0x82,0x0e,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  68 82 0e 20    andi    $s3, $a4, 0xe{{$}}
    andi    $s2, $t0, 0xf   # CHECK: andi $s2, $t0, 0xf # encoding: [0x4c,0x82,0x0f,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  4c 82 0f 20    andi    $s2, $t0, 0xf{{$}}
    andi    $a3, $s3, 12    # CHECK: andi $a3, $s3, 0xc # encoding: [0xf3,0x80,0x0c,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  f3 80 0c 20    andi    $a3, $s3, 0xc{{$}}
    andi    $a1, $s2, 13    # CHECK: andi $a1, $s2, 0xd # encoding: [0xb2,0x80,0x0d,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  b2 80 0d 20    andi    $a1, $s2, 0xd{{$}}
    andi    $a2, $s1, 16    # CHECK: andi $a2, $s1, 0x10    # encoding: [0xd1,0x80,0x10,0x20]
                            # CHECK-NEXT: # <MCInst #{{.*}} ANDI_NM
                            # DISAS: {{.*}}  d1 80 10 20    andi    $a2, $s1, 0x10{{$}}
    ori $a1, $s5, 0x1       # CHECK: ori $a1, $s5, 0x1  # encoding: [0xb5,0x80,0x01,0x00]
                            # CHECK-NEXT: # <MCInst #{{.*}} ORI_NM
                            # DISAS: {{.*}}  b5 80 01 00    ori $a1, $s5, 0x1{{$}}
    xori    $t1, $s2, 0xfff # CHECK: xori $t1, $s2, 0xfff   # encoding: [0xb2,0x81,0xff,0x1f]
                            # CHECK-NEXT: # <MCInst #{{.*}} XORI_NM
                            # DISAS: {{.*}}  b2 81 ff 1f    xori    $t1, $s2, 0xfff{{$}}

    # reg-imm bit-wise, 32-bit
    slti    $a1, $s2, 0xff  # CHECK: slti $a1, $s2, 255 # encoding: [0xb2,0x80,0xff,0x40]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLTI_NM
                            # DISAS: {{.*}}  b2 80 ff 40    slti    $a1, $s2, 255{{$}}
    sltiu   $a1, $s2, 0xfff # CHECK: sltiu $a1, $s2, 4095   # encoding: [0xb2,0x80,0xff,0x5f]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLTIU_NM
                            # DISAS: {{.*}}  b2 80 ff 5f    sltiu   $a1, $s2, 4095{{$}}
    seqi    $a1, $s2, 0xfff # CHECK: seqi $a1, $s2, 4095    # encoding: [0xb2,0x80,0xff,0x6f]
                            # CHECK-NEXT: # <MCInst #{{.*}} SEQI_NM
                            # DISAS: {{.*}}  b2 80 ff 6f    seqi    $a1, $s2, 4095{{$}}
    sll $a1, $s2, 9         # CHECK: sll $a1, $s2, 9    # encoding: [0xb2,0x80,0x09,0xc0]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLL_NM
                            # DISAS: {{.*}}  b2 80 09 c0    sll $a1, $s2, 9{{$}}
    sll $a1, $t2, 1         # CHECK: sll $a1, $t2, 1    # encoding: [0xae,0x80,0x01,0xc0]
                            # CHECK-NEXT: # <MCInst #{{.*}} SLL_NM
                            # DISAS: {{.*}}  ae 80 01 c0    sll $a1, $t2, 1{{$}}
    srl $a1, $a2, 31        # CHECK: srl $a1, $a2, 31   # encoding: [0xa6,0x80,0x5f,0xc0]
                            # CHECK-NEXT: # <MCInst #{{.*}} SRL_NM
                            # DISAS: {{.*}}  a6 80 5f c0    srl $a1, $a2, 31{{$}}
    srl $a1, $a7, 8         # CHECK: srl $a1, $a7, 8    # encoding: [0xab,0x80,0x48,0xc0]
                            # CHECK-NEXT: # <MCInst #{{.*}} SRL_NM
                            # DISAS: {{.*}}  ab 80 48 c0    srl $a1, $a7, 8{{$}}
    sra $a1, $a7, 15        # CHECK: sra $a1, $a7, 15   # encoding: [0xab,0x80,0x8f,0xc0]
                            # CHECK-NEXT: # <MCInst #{{.*}} SRA_NM
                            # DISAS: {{.*}}  ab 80 8f c0    sra $a1, $a7, 15{{$}}
    rotr    $a1, $s2, 31    # CHECK: rotr $a1, $s2, 31  # encoding: [0xb2,0x80,0xdf,0xc0]
                            # CHECK-NEXT: # <MCInst #{{.*}} ROTR_NM
                            # DISAS: {{.*}}  b2 80 df c0    rotr    $a1, $s2, 31{{$}}
    ext $a1, $s2, 0, 1      # CHECK: ext $a1, $s2, 0, 1 # encoding: [0xb2,0x80,0x00,0xf0]
                            # CHECK-NEXT: # <MCInst #{{.*}} EXT_NM
                            # DISAS: {{.*}}  b2 80 00 f0    ext $a1, $s2, 0, 1{{$}}
    ext $a1, $s2, 16, 15    # CHECK: ext $a1, $s2, 16, 15   # encoding: [0xb2,0x80,0x90,0xf3]
                            # CHECK-NEXT: # <MCInst #{{.*}} EXT_NM
                            # DISAS: {{.*}}  b2 80 90 f3    ext $a1, $s2, 16, 15{{$}}
    ext $a1, $s2, 0, 32     # CHECK: ext $a1, $s2, 0, 32    # encoding: [0xb2,0x80,0xc0,0xf7]
                            # CHECK-NEXT: # <MCInst #{{.*}} EXT_NM
                            # DISAS: {{.*}}  b2 80 c0 f7    ext $a1, $s2, 0, 32{{$}}
    ext $a1, $s2, 18, 17    # CHECK: ext $a1, $s2, 18, 17   # encoding: [0xb2,0x80,0x12,0xf4]
                            # CHECK-NEXT: # <MCInst #{{.*}} EXT_NM
                            # DISAS: {{.*}}  b2 80 12 f4    ext $a1, $s2, 18, 17{{$}}
    ins $a1, $s2, 31, 1     # CHECK: ins $a1, $s2, 31, 1    # encoding: [0xb2,0x80,0xdf,0xe7]
                            # CHECK-NEXT: # <MCInst #{{.*}} INS_NM
                            # DISAS: {{.*}}  b2 80 df e7    ins     $a1, $s2, 31, 1{{$}}
    ins $a1, $s2, 15, 16    # CHECK: ins $a1, $s2, 15, 16   # encoding: [0xb2,0x80,0x8f,0xe7]
                            # CHECK-NEXT: # <MCInst #{{.*}} INS_NM
                            # DISAS: {{.*}}  b2 80 8f e7    ins     $a1, $s2, 15, 16{{$}}
    ins $a1, $s2, 2, 30     # CHECK: ins $a1, $s2, 2, 30    # encoding: [0xb2,0x80,0xc2,0xe7]
                            # CHECK-NEXT: # <MCInst #{{.*}} INS_NM
                            # DISAS: {{.*}}  b2 80 c2 e7    ins     $a1, $s2, 2, 30{{$}}

    extw    $a0, $a1, $a2, 2    # CHECK: extw $a0, $a1, $a2, 2 # encoding: [0xc5,0x20,0x9f,0x20]
                                # CHECK-NEXT: # <MCInst #{{.*}} EXTW_NM
                                # DISAS: {{.*}}  c5 20 9f 20    extw    $a0, $a1, $a2, 2{{$}}
    extw    $a0, $s1, $t5, 0    # CHECK: extw $a0, $s1, $t5, 0  # encoding: [0x71,0x20,0x1f,0x20]
                                # CHECK-NEXT: # <MCInst #{{.*}} EXTW_NM
                                # DISAS: {{.*}}  71 20 1f 20    extw    $a0, $s1, $t5, 0{{$}}
    extw    $s1, $k1, $gp, 16   # CHECK: extw $s1, $k1, $gp, 16 # encoding: [0x9b,0x23,0x1f,0x8c]
                                # CHECK-NEXT: # <MCInst #{{.*}} EXTW_NM
                                # DISAS: {{.*}}  9b 23 1f 8c    extw    $s1, $k1, $gp, 16{{$}}
    extw    $a7, $s7, $t4, 31   # CHECK: extw $a7, $s7, $t4, 31 # encoding: [0x57,0x20,0xdf,0x5f]
                                # CHECK-NEXT: # <MCInst #{{.*}} EXTW_NM
                                # DISAS: {{.*}}  57 20 df 5f    extw    $a7, $s7, $t4, 31{{$}}
    rotx    $a1, $s3, 0, 0, 0   # CHECK: rotx $a1, $s3, 0, 0, 0 # encoding: [0xb3,0x80,0x00,0xd0]
                                # CHECK-NEXT: # <MCInst #{{.*}} ROTX_NM
                                # DISAS: {{.*}}  b3 80 00 d0    rotx    $a1, $s3, 0, 0, 0{{$}}
    rotx    $a7, $s4, 1, 2, 1   # CHECK: rotx $a7, $s4, 1, 2, 1 # encoding: [0x74,0x81,0xc1,0xd0]
                                # CHECK-NEXT: # <MCInst #{{.*}} ROTX_NM
                                # DISAS: {{.*}}  74 81 c1 d0    rotx    $a7, $s4, 1, 2, 1{{$}}
    rotx    $t8, $t0, 31, 6, 1  # CHECK: rotx $t8, $t0, 31, 6, 1 # encoding: [0x0c,0x83,0xdf,0xd1]
                                # CHECK-NEXT: # <MCInst #{{.*}} ROTX_NM
                                # DISAS: {{.*}}  0c 83 df d1    rotx    $t8, $t0, 31, 6, 1{{$}}
    rotx    $a1, $s3, 3, 16, 0  # CHECK: rotx $a1, $s3, 3, 16, 0 # encoding: [0xb3,0x80,0x03,0xd4]
                                # CHECK-NEXT: # <MCInst #{{.*}} ROTX_NM
                                # DISAS: {{.*}}  b3 80 03 d4    rotx    $a1, $s3, 3, 16, 0{{$}}
    rotx    $a7, $s4, 16, 8, 0  # CHECK: rotx $a7, $s4, 16, 8, 0 # encoding: [0x74,0x81,0x10,0xd2]
                                # CHECK-NEXT: # <MCInst #{{.*}} ROTX_NM
                                # DISAS: {{.*}}  74 81 10 d2    rotx    $a7, $s4, 16, 8, 0{{$}}
    rotx    $t8, $t0, 25, 30, 1 # CHECK: rotx $t8, $t0, 25, 30, 1 # encoding: [0x0c,0x83,0xd9,0xd7]
                                # CHECK-NEXT: # <MCInst #{{.*}} ROTX_NM
                                # DISAS: {{.*}}  0c 83 d9 d7    rotx    $t8, $t0, 25, 30, 1{{$}}
    bitrevw $t0, $t3            # CHECK: bitrevw $t0, $t3 # encoding: [0x8f,0x81,0x1f,0xd0]
                                # CHECK-NEXT: # <MCInst #{{.*}} BITREVW_NM
                                # DISAS: {{.*}}  8f 81 1f d0    bitrevw $t0, $t3
    byterevw $t0, $k0           # CHECK: byterevw $t0, $k0 # encoding: [0x9a,0x81,0x18,0xd2]
                                # CHECK-NEXT: # <MCInst #{{.*}} BYTEREVW_NM
                                # DISAS: {{.*}}  9a 81 18 d2    byterevw    $t0, $k0

    # compare and trap, 32-bit
    teq $s2, $a3, 0     # CHECK: teq $s2, $a3, 0    # encoding: [0xf2,0x20,0x00,0x00]
                        # CHECK-NEXT: # <MCInst #{{.*}} TEQ_NM
                        # DISAS: {{.*}}  f2 20 00 00    teq $s2, $a3, 0{{$}}
    teq $s2, $a3, 31    # CHECK: teq $s2, $a3, 31   # encoding: [0xf2,0x20,0x00,0xf8]
                        # CHECK-NEXT: # <MCInst #{{.*}} TEQ_NM
                        # DISAS: {{.*}}  f2 20 00 f8    teq $s2, $a3, 31{{$}}
    tne $s2, $a3, 15    # CHECK: tne $s2, $a3, 15   # encoding: [0xf2,0x20,0x00,0x7c]
                        # CHECK-NEXT: # <MCInst #{{.*}} TNE_NM
                        # DISAS: {{.*}}  f2 20 00 7c    tne $s2, $a3, 15{{$}}
    tne $s2, $a3, 1     # CHECK: tne $s2, $a3, 1    # encoding: [0xf2,0x20,0x00,0x0c]
                        # CHECK-NEXT: # <MCInst #{{.*}} TNE_NM
                        # DISAS: {{.*}}  f2 20 00 0c    tne $s2, $a3, 1{{$}}

    # single and paired moves
    move    $a4, $a7                # CHECK: move $a4, $a7  # encoding: [0x0b,0x11]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVE_NM
                                    # DISAS: {{.*}}  0b 11          move    $a4, $a7
    move    $s0, $k0                # CHECK: move $s0, $k0  # encoding: [0x1a,0x12]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVE_NM
                                    # DISAS: {{.*}}  1a 12          move    $s0, $k0
    move    $t4, $gp                # CHECK: move $t4, $gp  # encoding: [0x5c,0x10]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVE_NM
                                    # DISAS: {{.*}}  5c 10          move    $t4, $gp
    movep   $s0, $s1, $a0, $a1      # CHECK: movep $s0, $s1, $a0, $a1   # encoding: [0x30,0xfe]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEPREV_NM
                                    # DISAS: {{.*}}  30 fe          movep   $s0, $s1, $a0, $a1
    movep   $a7, $s7, $a1, $a2      # CHECK: movep $a7, $s7, $a1, $a2   # encoding: [0xe3,0xff]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEPREV_NM
                                    # DISAS: {{.*}}  e3 ff          movep   $a7, $s7, $a1, $a2
    movep   $a1, $s3, $a2, $a3      # CHECK: movep $a1, $s3, $a2, $a3   # encoding: [0x6d,0xfe]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEPREV_NM
                                    # DISAS: {{.*}}  6d fe          movep   $a1, $s3, $a2, $a3
    movep   $a6, $s0, $a3, $a4      # CHECK: movep $a6, $s0, $a3, $a4   # encoding: [0x0a,0xff]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEPREV_NM
                                    # DISAS: {{.*}}  0a ff          movep   $a6, $s0, $a3, $a4
    movep   $a0, $a1, $s0, $s1      # CHECK: movep $a0, $a1, $s0, $s1   # encoding: [0x30,0xbe]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEP_NM
                                    # DISAS: {{.*}}  30 be          movep   $a0, $a1, $s0, $s1
    movep   $a1, $a2, $zero, $s7    # CHECK: movep $a1, $a2, $zero, $s7 # encoding: [0xe3,0xbf]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEP_NM
                                    # DISAS: {{.*}}  e3 bf          movep   $a1, $a2, $zero, $s7
    movep   $a2, $a3, $a1, $s3      # CHECK: movep $a2, $a3, $a1, $s3   # encoding: [0x6d,0xbe]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEP_NM
                                    # DISAS: {{.*}}  6d be          movep   $a2, $a3, $a1, $s3
    movep   $a3, $a4, $a6, $s0      # CHECK: movep $a3, $a4, $a6, $s0   # encoding: [0x0a,0xbf]
                                    # CHECK-NEXT: # <MCInst #{{.*}} MOVEP_NM
                                    # DISAS: {{.*}}  0a bf          movep   $a3, $a4, $a6, $s0

    movn    $a1, $s2, $t4   # CHECK: movn $a1, $s2, $t4 # encoding: [0x52,0x20,0x10,0x2e]
                            # CHECK-NEXT: # <MCInst #{{.*}} MOVN_NM
                            # DISAS: {{.*}}  52 20 10 2e    movn    $a1, $s2, $t4
    movn    $a3, $s7, $t8   # CHECK: movn $a3, $s7, $t8 # encoding: [0x17,0x23,0x10,0x3e]
                            # CHECK-NEXT: # <MCInst #{{.*}} MOVN_NM
                            # DISAS: {{.*}}  17 23 10 3e    movn    $a3, $s7, $t8
    movn    $at, $t9, $s7   # CHECK: movn $at, $t9, $s7 # encoding: [0xf9,0x22,0x10,0x0e]
                            # CHECK-NEXT: # <MCInst #{{.*}} MOVN_NM
                            # DISAS: {{.*}}  f9 22 10 0e    movn    $at, $t9, $s7
    movz    $s1, $a2, $gp   # CHECK: movz $s1, $a2, $gp # encoding: [0x86,0x23,0x10,0x8a]
                            # CHECK-NEXT: # <MCInst #{{.*}} MOVZ_NM
                            # DISAS: {{.*}}  86 23 10 8a    movz    $s1, $a2, $gp
    movz    $s3, $a7, $sp   # CHECK: movz $s3, $a7, $sp # encoding: [0xab,0x23,0x10,0x9a]
                            # CHECK-NEXT: # <MCInst #{{.*}} MOVZ_NM
                            # DISAS: {{.*}}  ab 23 10 9a    movz    $s3, $a7, $sp
    movz    $s7, $k0, $fp   # CHECK: movz $s7, $k0, $fp # encoding: [0xda,0x23,0x10,0xba]
                            # CHECK-NEXT: # <MCInst #{{.*}} MOVZ_NM
                            # DISAS: {{.*}}  da 23 10 ba    movz    $s7, $k0, $fp

    clo $a0, $a1    # CHECK: clo $a0, $a1   # encoding: [0x85,0x20,0x3f,0x4b]
                    # CHECK-NEXT: # <MCInst #{{.*}} CLO_NM
                    # DISAS: {{.*}}  85 20 3f 4b    clo $a0, $a1
    clo $t0, $s1    # CHECK: clo $t0, $s1   # encoding: [0x91,0x21,0x3f,0x4b]
                    # CHECK-NEXT: # <MCInst #{{.*}} CLO_NM
                    # DISAS: {{.*}}  91 21 3f 4b    clo $t0, $s1
    clz $k0, $s7    # CHECK: clz $k0, $s7   # encoding: [0x57,0x23,0x3f,0x5b]
                    # CHECK-NEXT: # <MCInst #{{.*}} CLZ_NM
                    # DISAS: {{.*}}  57 23 3f 5b    clz $k0, $s7
    clz $s4, $a0    # CHECK: clz $s4, $a0   # encoding: [0x84,0x22,0x3f,0x5b]
                    # CHECK-NEXT: # <MCInst #{{.*}} CLZ_NM
                    # DISAS: {{.*}}  84 22 3f 5b    clz $s4, $a0

    seb $a0, $a1    # CHECK: seb $a0, $a1   # encoding: [0x85,0x20,0x08,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} SEB_NM
                    # DISAS: {{.*}}  85 20 08 00    seb $a0, $a1
    seb $t0, $s1    # CHECK: seb $t0, $s1   # encoding: [0x91,0x21,0x08,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} SEB_NM
                    # DISAS: {{.*}}  91 21 08 00    seb $t0, $s1
    seh $k0, $s7    # CHECK: seh $k0, $s7   # encoding: [0x57,0x23,0x48,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} SEH_NM
                    # DISAS: {{.*}}  57 23 48 00    seh $k0, $s7
    seh $s4, $a0    # CHECK: seh $s4, $a0   # encoding: [0x84,0x22,0x48,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} SEH_NM
                    # DISAS: {{.*}}  84 22 48 00    seh $s4, $a0

    lsa $a0, $a1, $a4, 1    # CHECK: lsa $a0, $a1, $a4, 1   # encoding: [0x05,0x21,0x0f,0x22]
                            # CHECK-NEXT: # <MCInst #{{.*}} LSA_NM
                            # DISAS: {{.*}}  05 21 0f 22    lsa $a0, $a1, $a4, 1{{$}}
    lsa $s0, $s1, $s4, 2    # CHECK: lsa $s0, $s1, $s4, 2   # encoding: [0x91,0x22,0x0f,0x84]
                            # CHECK-NEXT: # <MCInst #{{.*}} LSA_NM
                            # DISAS: {{.*}}  91 22 0f 84    lsa $s0, $s1, $s4, 2{{$}}
    lsa $s3, $a7, $a4, 3    # CHECK: lsa $s3, $a7, $a4, 3   # encoding: [0x0b,0x21,0x0f,0x9e]
                            # CHECK-NEXT: # <MCInst #{{.*}} LSA_NM
                            # DISAS: {{.*}}  0b 21 0f 9e    lsa $s3, $a7, $a4, 3{{$}}
    lsa $t4, $a3, $s4, 0    # CHECK: lsa $t4, $a3, $s4, 0   # encoding: [0x87,0x22,0x0f,0x10]
                            # CHECK-NEXT: # <MCInst #{{.*}} LSA_NM
                            # DISAS: {{.*}}  87 22 0f 10    lsa $t4, $a3, $s4, 0{{$}}

    di $a0      # CHECK: di $a0 # encoding: [0x80,0x20,0x7f,0x47]
                # CHECK-NEXT: # <MCInst #{{.*}} DI_NM
                # DISAS: {{.*}}  80 20 7f 47    di  $a0
    di $s7      # CHECK: di $s7 # encoding: [0xe0,0x22,0x7f,0x47]
                # CHECK-NEXT: # <MCInst #{{.*}} DI_NM
                # DISAS: {{.*}}  e0 22 7f 47    di  $s7
    ei $ra      # CHECK: ei $ra # encoding: [0xe0,0x23,0x7f,0x57]
                # CHECK-NEXT: # <MCInst #{{.*}} EI_NM
                # DISAS: {{.*}}  e0 23 7f 57    ei  $ra
    ei $t0      # CHECK: ei $t0 # encoding: [0x80,0x21,0x7f,0x57]
                # CHECK-NEXT: # <MCInst #{{.*}} EI_NM
                # DISAS: {{.*}}  80 21 7f 57    ei  $t0
    eret        # CHECK: eret   # encoding: [0x00,0x20,0x7f,0xf3]
                # CHECK-NEXT: # <MCInst #{{.*}} ERET_NM
                # DISAS: {{.*}}  00 20 7f f3    eret
    eretnc      # CHECK: eretnc # encoding: [0x01,0x20,0x7f,0xf3]
                # CHECK-NEXT: # <MCInst #{{.*}} ERETNC_NM
                # DISAS: {{.*}}  01 20 7f f3    eretnc
    deret       # CHECK: deret  # encoding: [0x00,0x20,0x7f,0xe3]
                # CHECK-NEXT: # <MCInst #{{.*}} DERET_NM
                # DISAS: {{.*}}  00 20 7f e3    deret
    nop32       # CHECK: nop32  # encoding: [0x00,0x80,0x00,0xc0]
                # CHECK-NEXT: # <MCInst #{{.*}} NOP32_NM
                # DISAS: {{.*}}  00 80 00 c0    nop32
    pause       # CHECK: pause  # encoding: [0x00,0x80,0x05,0xc0]
                # CHECK-NEXT: # <MCInst #{{.*}} PAUSE_NM
                # DISAS: {{.*}}  00 80 05 c0    pause
    ehb         # CHECK: ehb    # encoding: [0x00,0x80,0x03,0xc0]
                # CHECK-NEXT: # <MCInst #{{.*}} EHB_NM
                # DISAS: {{.*}}  00 80 03 c0    ehb

    sigrie 0        # CHECK: sigrie 0x0 # encoding: [0x00,0x00,0x00,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} SIGRIE_NM
                    # DISAS: {{.*}}  00 00 00 00    sigrie  0x0{{$}}
    sigrie 0xffff   # CHECK: sigrie 0xffff # encoding: [0x00,0x00,0xff,0xff]
                    # CHECK-NEXT: # <MCInst #{{.*}} SIGRIE_NM
                    # DISAS: {{.*}}  00 00 ff ff    sigrie  0xffff{{$}}
    sigrie 0x7ffff  # CHECK: sigrie 0x7ffff # encoding: [0x07,0x00,0xff,0xff]
                    # CHECK-NEXT: # <MCInst #{{.*}} SIGRIE_NM
                    # DISAS: {{.*}}  07 00 ff ff    sigrie  0x7ffff{{$}}
    sdbbp 0         # CHECK: sdbbp  0x0 # encoding: [0x18,0x10]
                    # CHECK-NEXT: # <MCInst #{{.*}} SDBBP16_NM
                    # DISAS: {{.*}}  18 10          sdbbp 0x0{{$}}
    sdbbp 7         # CHECK: sdbbp  0x7 # encoding: [0x1f,0x10]
                    # CHECK-NEXT: # <MCInst #{{.*}} SDBBP16_NM
                    # DISAS: {{.*}}  1f 10          sdbbp   0x7{{$}}
    sdbbp 8         # CHECK: sdbbp  0x8 # encoding: [0x18,0x00,0x08,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} SDBBP_NM
                    # DISAS: {{.*}}  18 00 08 00    sdbbp   0x8{{$}}
    sdbbp 0x7ffff   # CHECK: sdbbp  0x7ffff # encoding: [0x1f,0x00,0xff,0xff]
                    # CHECK-NEXT: # <MCInst #{{.*}} SDBBP_NM
                    # DISAS: {{.*}}  1f 00 ff ff    sdbbp   0x7ffff{{$}}
    break 0         # CHECK: break  0x0 # encoding: [0x10,0x10]
                    # CHECK-NEXT: # <MCInst #{{.*}} BREAK16_NM
                    # DISAS: {{.*}}  10 10          break   0x0{{$}}
    break 7         # CHECK: break  0x7 # encoding: [0x17,0x10]
                    # CHECK-NEXT: # <MCInst #{{.*}} BREAK16_NM
                    # DISAS: {{.*}}  17 10          break   0x7{{$}}
    break 8         # CHECK: break  0x8 # encoding: [0x10,0x00,0x08,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} BREAK_NM
                    # DISAS: {{.*}}  10 00 08 00    break   0x8{{$}}
    break 0x7ffff   # CHECK: break  0x7ffff # encoding: [0x17,0x00,0xff,0xff]
                    # CHECK-NEXT: # <MCInst #{{.*}} BREAK_NM
                    # DISAS: {{.*}}  17 00 ff ff    break   0x7ffff{{$}}
    syscall 0       # CHECK: syscall 0x0 # encoding: [0x08,0x10]
                    # CHECK-NEXT: # <MCInst #{{.*}} SYSCALL16_NM
                    # DISAS: {{.*}}  08 10          syscall 0x0{{$}}
    syscall 3       # CHECK: syscall 0x3 # encoding: [0x0b,0x10]
                    # CHECK-NEXT: # <MCInst #{{.*}} SYSCALL16_NM
                    # DISAS: {{.*}}  0b 10          syscall 0x3{{$}}
    syscall 4       # CHECK: syscall 0x4 # encoding: [0x08,0x00,0x04,0x00]
                    # CHECK-NEXT: # <MCInst #{{.*}} SYSCALL_NM
                    # DISAS: {{.*}}  08 00 04 00    syscall 0x4{{$}}
    syscall 0x3ffff # CHECK: syscall 0x3ffff # encoding: [0x0b,0x00,0xff,0xff]
                    # CHECK-NEXT: # <MCInst #{{.*}} SYSCALL_NM
                    # DISAS: {{.*}}  0b 00 ff ff    syscall 0x3ffff{{$}}
    rdpgpr $a3, $s7 # CHECK: rdpgpr $a3, $s7 # encoding: [0xf7,0x20,0x7f,0xe1]
                    # CHECK-NEXT: # <MCInst #{{.*}} RDPGPR_NM
                    # DISAS: {{.*}}  f7 20 7f e1    rdpgpr  $a3, $s7
    rdpgpr $fp, $sp # CHECK: rdpgpr $fp, $sp # encoding: [0xdd,0x23,0x7f,0xe1]
                    # CHECK-NEXT: # <MCInst #{{.*}} RDPGPR_NM
                    # DISAS: {{.*}}  dd 23 7f e1    rdpgpr  $fp, $sp
    wrpgpr $t0, $k0 # CHECK: wrpgpr $t0, $k0 # encoding: [0x9a,0x21,0x7f,0xf1]
                    # CHECK-NEXT: # <MCInst #{{.*}} WRPGPR_NM
                    # DISAS: {{.*}}  9a 21 7f f1    wrpgpr  $t0, $k0
    wrpgpr $a0, $ra # CHECK: wrpgpr $a0, $ra # encoding: [0x9f,0x20,0x7f,0xf1]
                    # CHECK-NEXT: # <MCInst #{{.*}} WRPGPR_NM
                    # DISAS: {{.*}}  9f 20 7f f1    wrpgpr  $a0, $ra

    rdhwr   $a0, $29, 0     # CHECK: rdhwr $a0, $29, 0  # encoding: [0x9d,0x20,0xc0,0x01]
                            # CHECK-NEXT: # <MCInst #{{.*}} RDHWR_NM
                            # DISAS: {{.*}}  9d 20 c0 01    rdhwr   $a0, $29, 0{{$}}

    # 16-bit SAVE/RESTORE[.JRC]
    save    32, $ra                     # CHECK: save 32, $ra # encoding: [0x21,0x1e]
                                        # CHECK-NEXT: # <MCInst #{{.*}} SAVE16_NM
                                        # DISAS: {{.*}}  21 1e          save    32, $ra
    save    32, $fp, $ra                # CHECK: save 32, $fp, $ra # encoding: [0x22,0x1c]
                                        # CHECK-NEXT: # <MCInst #{{.*}} SAVE16_NM
                                        # DISAS: {{.*}}  22 1c          save    32, $fp, $ra
    save    32, $ra, $s0                # CHECK: save 32, $ra, $s0  # encoding: [0x22,0x1e]
                                        # CHECK-NEXT: # <MCInst #{{.*}} SAVE16_NM
                                        # DISAS: {{.*}}  22 1e          save    32, $ra, $s0
    save    192, $ra, $s0-$gp           # CHECK: save 192, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0xce,0x1e]
                                        # CHECK-NEXT: # <MCInst #{{.*}} SAVE16_NM
                                        # DISAS: {{.*}}  ce 1e          save    192, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp
    save    192, $fp, $ra, $s0-$gp      # CHECK: save 192, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0xcf,0x1c]
                                        # CHECK-NEXT: # <MCInst #{{.*}} SAVE16_NM
                                        # DISAS: {{.*}}  cf 1c          save    192, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp
    save    128, $fp, $ra, $s0-$k1, $gp # CHECK: save 128, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0x8f,0x1c]
                                        # CHECK-NEXT: # <MCInst #{{.*}} SAVE16_NM
                                        # DISAS: {{.*}}  8f 1c          save    128, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp

    restore.jrc 32, $ra                     # CHECK: restore.jrc 32, $ra    # encoding: [0x21,0x1f]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  21 1f          restore.jrc 32, $ra
    restore.jrc 32, $fp, $ra                # CHECK: restore.jrc 32, $fp, $ra   # encoding: [0x22,0x1d]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  22 1d          restore.jrc 32, $fp, $ra
    restore.jrc 32, $ra, $s0                # CHECK: restore.jrc 32, $ra, $s0   # encoding: [0x22,0x1f]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  22 1f          restore.jrc 32, $ra, $s0
    restore.jrc 32, $ra, $s0-$s3            # CHECK: restore.jrc 32, $ra, $s0, $s1, $s2, $s3 # encoding: [0x25,0x1f]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  25 1f          restore.jrc 32, $ra, $s0, $s1, $s2, $s3
    restore.jrc 64, $fp, $ra, $s0-$s7       # CHECK: restore.jrc 64, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7 # encoding: [0x4a,0x1d]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  4a 1d          restore.jrc 64, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
    restore.jrc 192, $ra, $s0-$gp           # CHECK: restore.jrc 192, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0xce,0x1f]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  ce 1f          restore.jrc 192, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp
    restore.jrc 192, $fp, $ra, $s0-$gp      # CHECK: restore.jrc 192, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0xcf,0x1d]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  cf 1d          restore.jrc 192, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp
    restore.jrc 128, $fp, $ra, $s0-$k1, $gp # CHECK: restore.jrc 128, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0x8f,0x1d]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC16_NM
                                            # DISAS: {{.*}}  8f 1d          restore.jrc 128, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp

    # 32-bit SAVE/RESTORE[.JRC]
    save    40, $ra                         # CHECK: save 40, $ra   # encoding: [0xe1,0x83,0x28,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  e1 83 28 30    save    40, $ra
    save    48, $ra, $gp                    # CHECK: save 48, $ra, $gp  # encoding: [0xe2,0x83,0x34,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  e2 83 34 30    save    48, $ra, $gp
    save    256, $fp, $ra                   # CHECK: save 256, $fp, $ra # encoding: [0xc2,0x83,0x00,0x31]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  c2 83 00 31    save    256, $fp, $ra
    save    1024, $ra, $s0                  # CHECK: save 1024, $ra, $s0 # encoding: [0xe2,0x83,0x00,0x34]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  e2 83 00 34    save    1024, $ra, $s0
    save    160, $zero, $gp                 # CHECK: save 160, $zero, $gp # encoding: [0x02,0x80,0xa4,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  02 80 a4 30    save    160, $zero, $gp
    save    64, $s0-$s3, $gp                # CHECK: save 64, $s0, $s1, $s2, $s3, $gp # encoding: [0x05,0x82,0x44,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  05 82 44 30    save    64, $s0, $s1, $s2, $s3, $gp
    save    64, $a0-$s0, $gp                # CHECK: save 64, $a0, $a1, $a2, $a3, $a4, $a5, $a6, $a7, $t0, $t1, $t2, $t3, $s0, $gp # encoding: [0x8e,0x80,0x44,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  8e 80 44 30    save    64, $a0, $a1, $a2, $a3, $a4, $a5, $a6, $a7, $t0, $t1, $t2, $t3, $s0, $gp
    save    128, $t3, $s0, $s1, $gp         # CHECK: save 128, $t3, $s0, $s1, $gp # encoding: [0xe4,0x81,0x84,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  e4 81 84 30    save    128, $t3, $s0, $s1, $gp
    save    128, $a4-$t3, $s0-$s3           # CHECK: save 128, $a4, $a5, $a6, $a7, $t0, $t1, $t2, $t3, $s0, $s1, $s2, $s3 # encoding: [0x0c,0x81,0x80,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} SAVE_NM
                                            # DISAS: {{.*}}  0c 81 80 30    save    128, $a4, $a5, $a6, $a7, $t0, $t1, $t2, $t3, $s0, $s1, $s2, $s3
    restore 48, $ra                         # CHECK: restore 48, $ra    # encoding: [0xe1,0x83,0x32,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  e1 83 32 30    restore 48, $ra
    restore 56, $ra, $gp                    # CHECK: restore 56, $ra, $gp   # encoding: [0xe2,0x83,0x3e,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  e2 83 3e 30    restore 56, $ra, $gp
    restore 256, $fp, $ra                   # CHECK: restore 256, $fp, $ra  # encoding: [0xc2,0x83,0x02,0x31]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  c2 83 02 31    restore 256, $fp, $ra
    restore 1024, $ra, $s0                  # CHECK: restore 1024, $ra, $s0 # encoding: [0xe2,0x83,0x02,0x34]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  e2 83 02 34    restore 1024, $ra, $s0
    restore 160, $zero, $gp                 # CHECK: restore 160, $zero, $gp # encoding: [0x02,0x80,0xa6,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  02 80 a6 30    restore 160, $zero, $gp
    restore 64, $s0, $s1, $s2, $s3, $gp     # CHECK: restore 64,  $s0, $s1, $s2, $s3, $gp # encoding: [0x05,0x82,0x46,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  05 82 46 30    restore 64, $s0, $s1, $s2, $s3, $gp
    restore 64, $t4, $t5, $gp               # CHECK: restore 64, $t4, $t5, $gp # encoding: [0x43,0x80,0x46,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  43 80 46 30    restore 64, $t4, $t5, $gp
    restore 128, $fp, $ra, $s0, $s1, $gp    # CHECK: restore 128, $fp, $ra, $s0, $s1, $gp # encoding: [0xc5,0x83,0x86,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  c5 83 86 30    restore 128, $fp, $ra, $s0, $s1, $gp
    restore 120, $fp, $ra, $s0-$k1, $gp     # CHECK: restore 120, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0xcf,0x83,0x7e,0x30]
                                            # CHECK-NEXT: # <MCInst #{{.*}} RESTORE_NM
                                            # DISAS: {{.*}}  cf 83 7e 30    restore 120, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp

    restore.jrc 56, $ra                         # CHECK: restore.jrc 56, $ra    # encoding: [0xe1,0x83,0x3b,0x30]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  e1 83 3b 30    restore.jrc 56, $ra
    restore.jrc 256, $fp, $ra                   # CHECK: restore.jrc 256, $fp, $ra  # encoding: [0xc2,0x83,0x03,0x31]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  c2 83 03 31    restore.jrc 256, $fp, $ra
    restore.jrc 1024, $ra, $s0                  # CHECK: restore.jrc 1024, $ra, $s0 # encoding: [0xe2,0x83,0x03,0x34]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  e2 83 03 34    restore.jrc 1024, $ra, $s0
    restore.jrc 160, $zero, $gp                 # CHECK: restore.jrc 160, $zero, $gp    # encoding: [0x02,0x80,0xa7,0x30]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  02 80 a7 30    restore.jrc 160, $zero, $gp
    restore.jrc 64, $s0, $s1, $s2, $s3, $gp     # CHECK: restore.jrc 64, $s0, $s1, $s2, $s3, $gp # encoding: [0x05,0x82,0x47,0x30]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  05 82 47 30    restore.jrc 64, $s0, $s1, $s2, $s3, $gp
    restore.jrc 64, $t4, $t5, $gp               # CHECK: restore.jrc 64, $t4, $t5, $gp # encoding: [0x43,0x80,0x47,0x30]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  43 80 47 30    restore.jrc 64, $t4, $t5, $gp
    restore.jrc 128, $fp, $ra, $s0, $s1, $gp    # CHECK: restore.jrc 128, $fp, $ra, $s0, $s1, $gp # encoding: [0xc5,0x83,0x87,0x30]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  c5 83 87 30    restore.jrc 128, $fp, $ra, $s0, $s1, $gp
    restore.jrc 120, $fp, $ra, $s0-$k1, $gp     # CHECK: restore.jrc 120, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp # encoding: [0xcf,0x83,0x7f,0x30]
                                                # CHECK-NEXT: # <MCInst #{{.*}} RESTOREJRC_NM
                                                # DISAS: {{.*}}  cf 83 7f 30    restore.jrc 120, $fp, $ra, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $t8, $t9, $k0, $k1, $gp

    addiu   $a1,$a1,8           # CHECK: addiu  $a1, $a1, 8 # encoding: [0xd2,0x92]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIUR2_NM
                                # DISAS: {{.*}}  d2 92          addiu   $a1, $a1, 8{{$}}
    addiu   $a1,$a3,0           # CHECK: addiu  $a1, $a3, 0 # encoding: [0xf0,0x92]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIUR2_NM
                                # DISAS: {{.*}}  f0 92          addiu   $a1, $a3, 0{{$}}
    addiu   $s1,$a1,28          # CHECK: addiu  $s1, $a1, 28 # encoding: [0xd7,0x90]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIUR2_NM
                                # DISAS: {{.*}}  d7 90          addiu   $s1, $a1, 28{{$}}
    addiu   $a1,$a1,65535       # CHECK: addiu  $a1, $a1, 65535 # encoding: [0xa5,0x00,0xff,0xff]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIU_NM
                                # DISAS: {{.*}}  a5 00 ff ff    addiu   $a1, $a1, 65535{{$}}
    addiu   $s2,$s2,65536       # CHECK: addiu  $s2, $s2, 65536 # encoding: [0x41,0x62,0x00,0x00,0x01,0x00]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIU48_NM
                                # DISAS: {{.*}}  41 62 00 00 01 00      addiu   $s2, $s2, 65536{{$}}
    addiu   $t4,$t4,0x7fffffff  # CHECK: addiu  $t4, $t4, 2147483647 # encoding: [0x41,0x60,0xff,0xff,0xff,0x7f]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIU48_NM
                                # DISAS: {{.*}}  41 60 ff ff ff 7f      addiu   $t4, $t4, 2147483647{{$}}
    addiu   $a4,$a4,0           # CHECK: addiu  $a4, $a4, 0 # encoding: [0x08,0x91]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIURS5_NM
                                # DISAS: {{.*}}  08 91          addiu   $a4, $a4, 0{{$}}
    addiu   $t8,$t8,7           # CHECK: addiu  $t8, $t8, 7 # encoding: [0x0f,0x93]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIURS5_NM
                                # DISAS: {{.*}}  0f 93          addiu   $t8, $t8, 7{{$}}
    addiu   $sp,$sp,-8          # CHECK: addiu  $sp, $sp, -8 # encoding: [0xb8,0x93]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIURS5_NM
                                # DISAS: {{.*}}  b8 93          addiu   $sp, $sp, -8{{$}}
    addiu   $k1,$k1,8           # CHECK: addiu  $k1, $k1, 8 # encoding: [0x7b,0x03,0x08,0x00]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIU_NM
                                # DISAS: {{.*}}  7b 03 08 00    addiu   $k1, $k1, 8{{$}}
    addiu   $k1,$k1,-9          # CHECK: addiu  $k1, $k1, -9 # encoding: [0x7b,0x83,0x09,0x80]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIUNEG_NM
                                # DISAS: {{.*}}  7b 83 09 80    addiu   $k1, $k1, -9{{$}}
    addiu   $a1,$a2,-1          # CHECK: addiu  $a1, $a2, -1 # encoding: [0xa6,0x80,0x01,0x80]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIUNEG_NM
                                # DISAS: {{.*}}  a6 80 01 80    addiu   $a1, $a2, -1{{$}}
    addiu   $a1,$a2,-4095       # CHECK: addiu  $a1, $a2, -4095 # encoding: [0xa6,0x80,0xff,0x8f]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIUNEG_NM
                                # DISAS: {{.*}}  a6 80 ff 8f    addiu   $a1, $a2, -4095{{$}}
    addiu   $s3,$s3,-4096       # CHECK: addiu  $s3, $s3, -4096 # encoding: [0x61,0x62,0x00,0xf0,0xff,0xff]
                                # CHECK-NEXT: <MCInst  #{{.*}} ADDIU48_NM
                                # DISAS: {{.*}}  61 62 00 f0 ff ff      addiu   $s3, $s3, 4294963200{{$}}

    lui $s1, %hi(0x12345678)    # CHECK: lui $s1, %hi(0x12345000) # encoding: [0x34,0xe2,0x44,0x52]
                                # CHECK-NEXT: <MCInst #{{.*}} LUI_NM
                                # DISAS: {{.*}}  34 e2 44 52    lui $s1, %hi(0x12345000)
    lui $s2, %hi(0x80001000)    # CHECK: lui $s2, %hi(0x80001000) # encoding: [0x40,0xe2,0x01,0x10]
                                # CHECK-NEXT: <MCInst #{{.*}} LUI_NM
                                # DISAS: {{.*}}  40 e2 01 10    lui $s2, %hi(0x80001000)
    lui $s3, %hi(0xfffff000)    # CHECK: lui $s3, %hi(0xfffff000) # encoding: [0x7f,0xe2,0xfd,0xff]
                                # CHECK-NEXT: <MCInst #{{.*}} LUI_NM
                                # DISAS: {{.*}}  7f e2 fd ff    lui $s3, %hi(0xfffff000)

    brsc    $a1                 # CHECK: brsc $a1   # encoding: [0x05,0x48,0x00,0x80]
                                # CHECK-NEXT: # <MCInst #{{.*}} BRSC_NM
                                # DISAS: {{.*}}  05 48 00 80    brsc    $a1
    brsc    $a3                 # CHECK: brsc $a3   # encoding: [0x07,0x48,0x00,0x80]
                                # CHECK-NEXT: # <MCInst #{{.*}} BRSC_NM
                                # DISAS: {{.*}}  07 48 00 80    brsc    $a3
    balrsc  $ra, $t0            # CHECK: balrsc $ra, $t0    # encoding: [0xec,0x4b,0x00,0x80]
                                # CHECK-NEXT: # <MCInst #{{.*}} BALRSC_NM
                                # DISAS: {{.*}}  ec 4b 00 80    balrsc  $ra, $t0
    balrsc  $sp, $gp            # CHECK: balrsc $sp, $gp    # encoding: [0xbc,0x4b,0x00,0x80]
                                # CHECK-NEXT: # <MCInst #{{.*}} BALRSC_NM
                                # DISAS: {{.*}}  bc 4b 00 80    balrsc  $sp, $gp

    jalrc   $t2, $s2    # CHECK: jalrc $t2, $s2 # encoding: [0xd2,0x49,0x00,0x00]
                        # CHECK-NEXT: # <MCInst #{{.*}} JALRC_NM
                        # DISAS: {{.*}}  d2 49 00 00    jalrc   $t2, $s2
    jalrc   $a5, $s5    # CHECK: jalrc $a5, $s5 # encoding: [0x35,0x49,0x00,0x00]
                        # CHECK-NEXT: # <MCInst #{{.*}} JALRC_NM
                        # DISAS: {{.*}}  35 49 00 00    jalrc   $a5, $s5
    jalrc.hb $t0, $t1   # CHECK: jalrc.hb $t0, $t1  # encoding: [0x8d,0x49,0x00,0x10]
                        # CHECK-NEXT: # <MCInst #{{.*}} JALRCHB_NM
                        # DISAS: {{.*}}  8d 49 00 10    jalrc.hb    $t0, $t1
    jalrc.hb $s5, $a5   # CHECK: jalrc.hb $s5, $a5  # encoding: [0xa9,0x4a,0x00,0x10]
                        # CHECK-NEXT: # <MCInst #{{.*}} JALRCHB_NM
                        # DISAS: {{.*}}  a9 4a 00 10    jalrc.hb    $s5, $a5
    jrc $a1             # CHECK: jrc $a1    # encoding: [0xa0,0xd8]
                        # CHECK-NEXT: # <MCInst #{{.*}} JRC_NM
                        # DISAS: {{.*}}  a0 d8          jrc $a1
    jrc $ra             # CHECK: jrc $ra    # encoding: [0xe0,0xdb]
                        # CHECK-NEXT: # <MCInst #{{.*}} JRC_NM
                        # DISAS: {{.*}}  e0 db          jrc $ra

    wait 1          # CHECK: wait 1 # encoding: [0x01,0x20,0x7f,0xc3]
                    # CHECK-NEXT: # <MCInst #{{.*}} WAIT_NM
                    # DISAS: {{.*}}  01 20 7f c3    wait    1{{$}}
    wait 1023       # CHECK: wait 1023  # encoding: [0xff,0x23,0x7f,0xc3]
                    # CHECK-NEXT: # <MCInst #{{.*}} WAIT_NM
                    # DISAS: {{.*}}  ff 23 7f c3    wait    1023{{$}}

    evp $a1         # CHECK: evp $a1    # encoding: [0xa0,0x20,0x90,0x07]
                    # CHECK-NEXT: <MCInst #{{.*}} EVP_NM
                    # DISAS: {{.*}}  a0 20 90 07    evp $a1
    dvp $a1         # CHECK: dvp $a1    # encoding: [0xa0,0x20,0x90,0x03]
                    # CHECK-NEXT: <MCInst #{{.*}} DVP_NM
                    # DISAS: {{.*}}  a0 20 90 03    dvp $a1

    jrc $ra
    .type   g_8,@object
    .comm   g_8,16,16
