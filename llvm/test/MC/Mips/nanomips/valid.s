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

    evp $a1         # CHECK: evp $a1    # encoding: [0xa0,0x20,0x90,0x07]
                    # CHECK-NEXT: <MCInst #{{.*}} EVP_NM
                    # DISAS: {{.*}}  a0 20 90 07    evp $a1
    dvp $a1         # CHECK: dvp $a1    # encoding: [0xa0,0x20,0x90,0x03]
                    # CHECK-NEXT: <MCInst #{{.*}} DVP_NM
                    # DISAS: {{.*}}  a0 20 90 03    dvp $a1

    jrc $ra
    .type   g_8,@object
    .comm   g_8,16,16
