# Instructions that are valid
#
# Branches have some unusual encoding rules in MIPS32r6 so we need to test:
#   rs == 0
#   rs != 0
#   rt == 0
#   rt != 0
#   rs < rt
#   rs == rt
#   rs > rt
# appropriately for each branch instruction
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -show-inst -mcpu=mips64r6 2> %t0 | FileCheck %s
# RUN: FileCheck %s -check-prefix=WARNING < %t0
a:
        .set noat
        # FIXME: Add the instructions carried forward from older ISA's
        addiupc $4, 100          # CHECK: lapc $4, 100        # encoding: [0xec,0x80,0x00,0x19]
        addu    $9,10            # CHECK: addiu $9, $9, 10    # encoding: [0x25,0x29,0x00,0x0a]
        align   $4, $2, $3, 2    # CHECK: align $4, $2, $3, 2 # encoding: [0x7c,0x43,0x22,0xa0]
        aluipc  $3, 56           # CHECK: aluipc $3, 56       # encoding: [0xec,0x7f,0x00,0x38]
        and     $2,4             # CHECK: andi $2, $2, 4      # encoding: [0x30,0x42,0x00,0x04]
        aui     $3, $2, 23       # CHECK: aui $3, $2, 23      # encoding: [0x3c,0x43,0x00,0x17]
        auipc   $3, -1           # CHECK: auipc $3, -1        # encoding: [0xec,0x7e,0xff,0xff]
        bal     21100            # CHECK: bal 21100           # encoding: [0x04,0x11,0x14,0x9b]
        balc 14572256            # CHECK: balc 14572256       # encoding: [0xe8,0x37,0x96,0xb8]
        bc 14572256              # CHECK: bc 14572256         # encoding: [0xc8,0x37,0x96,0xb8]
        bc1eqz  $f0,4            # CHECK: bc1eqz $f0, 4       # encoding: [0x45,0x20,0x00,0x01]
        bc1eqz  $f31,4           # CHECK: bc1eqz $f31, 4      # encoding: [0x45,0x3f,0x00,0x01]
        bc1nez  $f0,4            # CHECK: bc1nez $f0, 4       # encoding: [0x45,0xa0,0x00,0x01]
        bc1nez  $f31,4           # CHECK: bc1nez $f31, 4      # encoding: [0x45,0xbf,0x00,0x01]
        bc2eqz  $0,8             # CHECK: bc2eqz $0, 8        # encoding: [0x49,0x20,0x00,0x02]
        bc2eqz  $31,8            # CHECK: bc2eqz $31, 8       # encoding: [0x49,0x3f,0x00,0x02]
        bc2nez  $0,8             # CHECK: bc2nez $0, 8        # encoding: [0x49,0xa0,0x00,0x02]
        bc2nez  $31,8            # CHECK: bc2nez $31, 8       # encoding: [0x49,0xbf,0x00,0x02]
        # beqc requires rs < rt && rs != 0 but we accept this and fix it. See also bovc.
        beqc    $5, $6, 256      # CHECK: beqc $5, $6, 256    # encoding: [0x20,0xa6,0x00,0x40]
        beqc    $6, $5, 256      # CHECK: beqc $6, $5, 256    # encoding: [0x20,0xa6,0x00,0x40]
        beqzalc $2, 1332         # CHECK: beqzalc $2, 1332    # encoding: [0x20,0x02,0x01,0x4d]
        beqzc $5, 72256          # CHECK: beqzc $5, 72256     # encoding: [0xd8,0xa0,0x46,0x90]
        bgec $2, $3, 256         # CHECK: bgec $2, $3, 256    # encoding: [0x58,0x43,0x00,0x40]
        bgeuc $2, $3, 256        # CHECK: bgeuc $2, $3, 256   # encoding: [0x18,0x43,0x00,0x40]
        bgezalc $2, 1332         # CHECK: bgezalc $2, 1332    # encoding: [0x18,0x42,0x01,0x4d]
        bgezc $5, 256            # CHECK: bgezc $5, 256       # encoding: [0x58,0xa5,0x00,0x40]
        bgtzalc $2, 1332         # CHECK: bgtzalc $2, 1332    # encoding: [0x1c,0x02,0x01,0x4d]
        bgtzc $5, 256            # CHECK: bgtzc $5, 256       # encoding: [0x5c,0x05,0x00,0x40]
        bitswap $4, $2           # CHECK: bitswap $4, $2      # encoding: [0x7c,0x02,0x20,0x20]
        blezalc $2, 1332         # CHECK: blezalc $2, 1332    # encoding: [0x18,0x02,0x01,0x4d]
        blezc $5, 256            # CHECK: blezc $5, 256       # encoding: [0x58,0x05,0x00,0x40]
        bltc $5, $6, 256         # CHECK: bltc $5, $6, 256    # encoding: [0x5c,0xa6,0x00,0x40]
        bltuc $5, $6, 256        # CHECK: bltuc $5, $6, 256   # encoding: [0x1c,0xa6,0x00,0x40]
        bltzalc $2, 1332         # CHECK: bltzalc $2, 1332    # encoding: [0x1c,0x42,0x01,0x4d]
        bltzc $5, 256            # CHECK: bltzc $5, 256       # encoding: [0x5c,0xa5,0x00,0x40]
        # bnec requires rs < rt && rs != 0 but we accept this and fix it. See also bnvc.
        bnec $5, $6, 256         # CHECK: bnec $5, $6, 256    # encoding: [0x60,0xa6,0x00,0x40]
        bnec $6, $5, 256         # CHECK: bnec $6, $5, 256    # encoding: [0x60,0xa6,0x00,0x40]
        bnezalc $2, 1332         # CHECK: bnezalc $2, 1332    # encoding: [0x60,0x02,0x01,0x4d]
        bnezc $5, 72256          # CHECK: bnezc $5, 72256     # encoding: [0xf8,0xa0,0x46,0x90]
        # bnvc requires that rs >= rt but we accept both and fix this. See also bnec.
        bnvc     $0, $0, 4       # CHECK: bnvc $zero, $zero, 4 # encoding: [0x60,0x00,0x00,0x01]
        bnvc     $2, $0, 4       # CHECK: bnvc $2, $zero, 4    # encoding: [0x60,0x40,0x00,0x01]
        bnvc     $2, $4, 4       # CHECK: bnvc $2, $4, 4      # encoding: [0x60,0x82,0x00,0x01]
        # bovc requires that rs >= rt but we accept both and fix this. See also beqc.
        bovc     $0, $0, 4       # CHECK: bovc $zero, $zero, 4 # encoding: [0x20,0x00,0x00,0x01]
        bovc     $2, $0, 4       # CHECK: bovc $2, $zero, 4    # encoding: [0x20,0x40,0x00,0x01]
        bovc     $2, $4, 4       # CHECK: bovc $2, $4, 4      # encoding: [0x20,0x82,0x00,0x01]
        cache    1, 8($5)        # CHECK: cache 1, 8($5)         # encoding: [0x7c,0xa1,0x04,0x25]
        class.d $f2, $f4         # CHECK: class.d $f2, $f4       # encoding: [0x46,0x20,0x20,0x9b]
        class.s $f2, $f4         # CHECK: class.s $f2, $f4       # encoding: [0x46,0x00,0x20,0x9b]
        clo     $11,$a1          # CHECK: clo $11, $5            # encoding: [0x00,0xa0,0x58,0x51]
        clz     $sp,$gp          # CHECK: clz $sp, $gp           # encoding: [0x03,0x80,0xe8,0x50]
        ceil.w.d  $f11,$f25         # CHECK: ceil.w.d  $f11, $f25 # encoding: [0x46,0x20,0xca,0xce]
                                    # CHECK:                      # <MCInst #{{.*}} CEIL_W_D64
        ceil.w.s  $f6,$f20          # CHECK: ceil.w.s  $f6, $f20  # encoding: [0x46,0x00,0xa1,0x8e]
                                    # CHECK:                      # <MCInst #{{.*}} CEIL_W_S
        cmp.af.d   $f2,$f3,$f4      # CHECK: cmp.af.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x80]
        cmp.af.s   $f2,$f3,$f4      # CHECK: cmp.af.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x80]
        cmp.eq.d   $f2,$f3,$f4      # CHECK: cmp.eq.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x82]
        cmp.eq.s   $f2,$f3,$f4      # CHECK: cmp.eq.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x82]
        cmp.le.d   $f2,$f3,$f4      # CHECK: cmp.le.d  $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x86]
        cmp.le.s   $f2,$f3,$f4      # CHECK: cmp.le.s  $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x86]
        cmp.lt.d   $f2,$f3,$f4      # CHECK: cmp.lt.d  $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x84]
        cmp.lt.s   $f2,$f3,$f4      # CHECK: cmp.lt.s  $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x84]
        cmp.saf.d  $f2,$f3,$f4      # CHECK: cmp.saf.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x88]
        cmp.saf.s  $f2,$f3,$f4      # CHECK: cmp.saf.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x88]
        cmp.seq.d  $f2,$f3,$f4      # CHECK: cmp.seq.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x8a]
        cmp.seq.s  $f2,$f3,$f4      # CHECK: cmp.seq.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x8a]
        cmp.sle.d  $f2,$f3,$f4      # CHECK: cmp.sle.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x8e]
        cmp.sle.s  $f2,$f3,$f4      # CHECK: cmp.sle.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x8e]
        cmp.slt.d  $f2,$f3,$f4      # CHECK: cmp.slt.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x8c]
        cmp.slt.s  $f2,$f3,$f4      # CHECK: cmp.slt.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x8c]
        cmp.sueq.d $f2,$f3,$f4      # CHECK: cmp.sueq.d $f2, $f3, $f4 # encoding: [0x46,0xa4,0x18,0x8b]
        cmp.sueq.s $f2,$f3,$f4      # CHECK: cmp.sueq.s $f2, $f3, $f4 # encoding: [0x46,0x84,0x18,0x8b]
        cmp.sule.d $f2,$f3,$f4      # CHECK: cmp.sule.d $f2, $f3, $f4 # encoding: [0x46,0xa4,0x18,0x8f]
        cmp.sule.s $f2,$f3,$f4      # CHECK: cmp.sule.s $f2, $f3, $f4 # encoding: [0x46,0x84,0x18,0x8f]
        cmp.sult.d $f2,$f3,$f4      # CHECK: cmp.sult.d $f2, $f3, $f4 # encoding: [0x46,0xa4,0x18,0x8d]
        cmp.sult.s $f2,$f3,$f4      # CHECK: cmp.sult.s $f2, $f3, $f4 # encoding: [0x46,0x84,0x18,0x8d]
        cmp.sun.d  $f2,$f3,$f4      # CHECK: cmp.sun.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x89]
        cmp.sun.s  $f2,$f3,$f4      # CHECK: cmp.sun.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x89]
        cmp.ueq.d  $f2,$f3,$f4      # CHECK: cmp.ueq.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x83]
        cmp.ueq.s  $f2,$f3,$f4      # CHECK: cmp.ueq.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x83]
        cmp.ule.d  $f2,$f3,$f4      # CHECK: cmp.ule.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x87]
        cmp.ule.s  $f2,$f3,$f4      # CHECK: cmp.ule.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x87]
        cmp.ult.d  $f2,$f3,$f4      # CHECK: cmp.ult.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x85]
        cmp.ult.s  $f2,$f3,$f4      # CHECK: cmp.ult.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x85]
        cmp.un.d   $f2,$f3,$f4      # CHECK: cmp.un.d $f2, $f3, $f4  # encoding: [0x46,0xa4,0x18,0x81]
        cmp.un.s   $f2,$f3,$f4      # CHECK: cmp.un.s $f2, $f3, $f4  # encoding: [0x46,0x84,0x18,0x81]
        daddu   $19,26943        # CHECK: daddiu $19, $19, 26943 # encoding: [0x66,0x73,0x69,0x3f]
        daddu   $24,$2,18079     # CHECK: daddiu $24, $2, 18079  # encoding: [0x64,0x58,0x46,0x9f]
        dahi    $3, $3, 0x5678   # CHECK: dahi $3, $3, 22136     # encoding: [0x04,0x66,0x56,0x78]
        dalign  $4,$2,$3,5       # CHECK: dalign $4, $2, $3, 5 # encoding: [0x7c,0x43,0x23,0x64]
        dati     $3, $3, 0xabcd  # CHECK: dati $3, $3, 43981   # encoding: [0x04,0x7e,0xab,0xcd]
        daui     $3, $2, 0x1234  # CHECK: daui $3, $2, 4660    # encoding: [0x74,0x43,0x12,0x34]
        dbitswap $4, $2          # CHECK: dbitswap $4, $2    # encoding: [0x7c,0x02,0x20,0x24]
        dclo    $s2,$a2          # CHECK: dclo $18, $6           # encoding: [0x00,0xc0,0x90,0x53]
        dclz    $s0,$25          # CHECK: dclz $16, $25          # encoding: [0x03,0x20,0x80,0x52]
        ddiv    $2,$3,$4         # CHECK: ddiv $2, $3, $4  # encoding: [0x00,0x64,0x10,0x9e]
        ddivu   $2,$3,$4         # CHECK: ddivu $2, $3, $4 # encoding: [0x00,0x64,0x10,0x9f]
        dext    $9,$6,3,7        # CHECK: dext $9, $6, 3, 7      # encoding: [0x7c,0xc9,0x30,0xc3]
        dextm   $9,$6,3,39       # CHECK: dextm $9, $6, 3, 39    # encoding: [0x7c,0xc9,0x30,0xc1]
        dextu   $9,$6,35,7       # CHECK: dextu $9, $6, 35, 7    # encoding: [0x7c,0xc9,0x30,0xc2]
        di                       # CHECK: di               # encoding: [0x41,0x60,0x60,0x00]
        di      $s8              # CHECK: di  $fp          # encoding: [0x41,0x7e,0x60,0x00]
        dins    $2,$3,4,28       # CHECK: dins  $2, $3, 4, 28    # encoding: [0x7c,0x62,0xf9,0x07]
        dinsm   $2,$3,4,34       # CHECK: dinsm $2, $3, 4, 34    # encoding: [0x7c,0x62,0x29,0x05]
        dinsu   $2,$3,34,16      # CHECK: dinsu $2, $3, 34, 16   # encoding: [0x7c,0x62,0x88,0x86]
        div     $2,$3,$4         # CHECK: div $2, $3, $4   # encoding: [0x00,0x64,0x10,0x9a]
        divu    $2,$3,$4         # CHECK: divu $2, $3, $4  # encoding: [0x00,0x64,0x10,0x9b]
        dlsa    $2, $3, $4, 3    # CHECK: dlsa $2, $3, $4, 3 # encoding: [0x00,0x64,0x10,0x95]
        dmfc0   $10, $16, 2      # CHECK: dmfc0 $10, $16, 2  # encoding: [0x40,0x2a,0x80,0x02]
        dmod    $2,$3,$4         # CHECK: dmod $2, $3, $4  # encoding: [0x00,0x64,0x10,0xde]
        dmodu   $2,$3,$4         # CHECK: dmodu $2, $3, $4 # encoding: [0x00,0x64,0x10,0xdf]
        dmtc0   $4, $10, 0       # CHECK: dmtc0 $4, $10, 0 # encoding: [0x40,0xa4,0x50,0x00]
        dmuh    $2,$3,$4         # CHECK: dmuh $2, $3, $4  # encoding: [0x00,0x64,0x10,0xdc]
        dmuhu   $2,$3,$4         # CHECK: dmuhu $2, $3, $4 # encoding: [0x00,0x64,0x10,0xdd]
        dmul    $2,$3,$4         # CHECK: dmul $2, $3, $4  # encoding: [0x00,0x64,0x10,0x9c]
        dmulu   $2,$3,$4         # CHECK: dmulu $2, $3, $4 # encoding: [0x00,0x64,0x10,0x9d]
        dneg      $2             # CHECK: dneg $2, $2      # encoding: [0x00,0x02,0x10,0x2e]
        dneg      $2,$3          # CHECK: dneg $2, $3      # encoding: [0x00,0x03,0x10,0x2e]
        dnegu     $2,$3          # CHECK: dnegu $2, $3     # encoding: [0x00,0x03,0x10,0x2f]
        dsll      $4, $5         # CHECK: dsllv $4, $4, $5 # encoding: [0x00,0xa4,0x20,0x14]
        dsll      $4, $5, $5     # CHECK: dsllv $4, $5, $5 # encoding: [0x00,0xa5,0x20,0x14]
        dsrl      $4, $5         # CHECK: dsrlv $4, $4, $5 # encoding: [0x00,0xa4,0x20,0x16]
        dsrl      $4, $4, $5     # CHECK: dsrlv $4, $4, $5 # encoding: [0x00,0xa4,0x20,0x16]
        dsubu   $14,-4586        # CHECK: daddiu $14, $14, 4586  # encoding: [0x65,0xce,0x11,0xea]
        dsubu   $15,$11,5025     # CHECK: daddiu $15, $11, -5025 # encoding: [0x65,0x6f,0xec,0x5f]
        dvp     $4               # CHECK: dvp $4           # encoding: [0x41,0x64,0x00,0x24]
        dvp                      # CHECK: dvp $zero        # encoding: [0x41,0x60,0x00,0x24]
        ei                       # CHECK: ei               # encoding: [0x41,0x60,0x60,0x20]
        ei      $14              # CHECK: ei  $14          # encoding: [0x41,0x6e,0x60,0x20]
        eretnc                   # CHECK: eretnc                 # encoding: [0x42,0x00,0x00,0x58]
        evp     $5               # CHECK: evp $5           # encoding: [0x41,0x65,0x00,0x04]
        evp                      # CHECK: evp $zero        # encoding: [0x41,0x60,0x00,0x04]
        floor.w.d $f14,$f11      # CHECK: floor.w.d $f14, $f11      # encoding: [0x46,0x20,0x5b,0x8f]
                                 # CHECK:                           # <MCInst #{{.*}} FLOOR_W_D64
        floor.w.s $f8,$f9        # CHECK: floor.w.s $f8, $f9        # encoding: [0x46,0x00,0x4a,0x0f]
                                 # CHECK:                           # <MCInst #{{.*}} FLOOR_W_S
        j       1f               # CHECK: j $tmp0                # encoding: [0b000010AA,A,A,A]
                                 # CHECK:                        #   fixup A - offset: 0, value: ($tmp0), kind: fixup_Mips_26
        j       a                # CHECK: j a                    # encoding: [0b000010AA,A,A,A]
                                 # CHECK:                        #   fixup A - offset: 0, value: a, kind: fixup_Mips_26
        j       1328             # CHECK: j 1328                 # encoding: [0x08,0x00,0x01,0x4c]
        jal     21100            # CHECK: jal 21100              # encoding: [0x0c,0x00,0x14,0x9b]
        jr.hb   $4               # CHECK: jr.hb $4               # encoding: [0x00,0x80,0x04,0x09]
        jr      $ra              # CHECK: jr $ra                 # encoding: [0x03,0xe0,0x00,0x09]
        jr      $25              # CHECK: jr $25                 # encoding: [0x03,0x20,0x00,0x09]        
        jrc     $27              # CHECK: jrc $27                # encoding: [0xd8,0x1b,0x00,0x00]
        jalr.hb $4               # CHECK: jalr.hb $4             # encoding: [0x00,0x80,0xfc,0x09]
        jalr.hb $4, $5           # CHECK: jalr.hb $4, $5         # encoding: [0x00,0xa0,0x24,0x09]
        jalrc   $25              # CHECK: jalrc $25              # encoding: [0xf8,0x19,0x00,0x00]
        jialc   $5, 256          # CHECK: jialc $5, 256    # encoding: [0xf8,0x05,0x01,0x00]
        jic     $5, 256          # CHECK: jic $5, 256      # encoding: [0xd8,0x05,0x01,0x00]
        l.s     $f2, 8($3)       # CHECK: lwc1  $f2, 8($3)       # encoding: [0xc4,0x62,0x00,0x08]
        l.d     $f2, 8($3)       # CHECK: ldc1  $f2, 8($3)       # encoding: [0xd4,0x62,0x00,0x08]
        lapc    $4, 100          # CHECK: lapc $4, 100           # encoding: [0xec,0x80,0x00,0x19]
        ldc2    $8, -701($at)    # CHECK: ldc2 $8, -701($1)      # encoding: [0x49,0xc8,0x0d,0x43]
        ldpc    $2,123456        # CHECK: ldpc $2, 123456  # encoding: [0xec,0x58,0x3c,0x48]
        ll      $v0,-153($s2)    # CHECK: ll $2, -153($18)       # encoding: [0x7e,0x42,0xb3,0xb6]
        lld     $zero,112($ra)   # CHECK: lld $zero, 112($ra)    # encoding: [0x7f,0xe0,0x38,0x37]
        lsa     $2, $3, $4, 3    # CHECK: lsa  $2, $3, $4, 3     # encoding: [0x00,0x64,0x10,0x85]
        lwc2    $18,-841($a2)    # CHECK: lwc2 $18, -841($6)     # encoding: [0x49,0x52,0x34,0xb7]
        lwpc    $2,268           # CHECK: lwpc $2, 268     # encoding: [0xec,0x48,0x00,0x43]
        lwupc   $2,268           # CHECK: lwupc $2, 268    # encoding: [0xec,0x50,0x00,0x43]
        maddf.d $f2,$f3,$f4      # CHECK: maddf.d $f2, $f3, $f4  # encoding: [0x46,0x24,0x18,0x98]
        maddf.s $f2,$f3,$f4      # CHECK: maddf.s $f2, $f3, $f4  # encoding: [0x46,0x04,0x18,0x98]
        max.d   $f0, $f2, $f4    # CHECK: max.d $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x1e]
        max.s   $f0, $f2, $f4    # CHECK: max.s $f0, $f2, $f4 # encoding: [0x46,0x04,0x10,0x1e]
        maxa.d  $f0, $f2, $f4    # CHECK: maxa.d $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x1f]
        maxa.s  $f0, $f2, $f4    # CHECK: maxa.s $f0, $f2, $f4 # encoding: [0x46,0x04,0x10,0x1f]
        min.d   $f0, $f2, $f4    # CHECK: min.d $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x1c]
        min.s   $f0, $f2, $f4    # CHECK: min.s $f0, $f2, $f4 # encoding: [0x46,0x04,0x10,0x1c]
        mina.d  $f0, $f2, $f4    # CHECK: mina.d $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x1d]
        mina.s  $f0, $f2, $f4    # CHECK: mina.s $f0, $f2, $f4 # encoding: [0x46,0x04,0x10,0x1d]
        mfc0    $8,$15,1         # CHECK: mfc0 $8, $15, 1      # encoding: [0x40,0x08,0x78,0x01]
        mod     $2,$3,$4         # CHECK: mod $2, $3, $4   # encoding: [0x00,0x64,0x10,0xda]
        modu    $2,$3,$4         # CHECK: modu $2, $3, $4  # encoding: [0x00,0x64,0x10,0xdb]
        move    $a0,$a3          # CHECK: move $4, $7             # encoding: [0x00,0xe0,0x20,0x25]
        move    $s5,$a0          # CHECK: move $21, $4            # encoding: [0x00,0x80,0xa8,0x25]
        move    $s8,$a0          # CHECK: move $fp, $4            # encoding: [0x00,0x80,0xf0,0x25]
        move    $25,$a2          # CHECK: move $25, $6            # encoding: [0x00,0xc0,0xc8,0x25]
        mtc0    $9,$15,1         # CHECK: mtc0 $9, $15, 1        # encoding: [0x40,0x89,0x78,0x01]
        msubf.d $f2,$f3,$f4      # CHECK: msubf.d $f2, $f3, $f4  # encoding: [0x46,0x24,0x18,0x99]
        msubf.s $f2,$f3,$f4      # CHECK: msubf.s $f2, $f3, $f4  # encoding: [0x46,0x04,0x18,0x99]
        muh     $2,$3,$4         # CHECK: muh $2, $3, $4   # encoding: [0x00,0x64,0x10,0xd8]
        muhu    $2,$3,$4         # CHECK: muhu $2, $3, $4  # encoding: [0x00,0x64,0x10,0xd9]
        mul     $2,$3,$4         # CHECK: mul $2, $3, $4   # encoding: [0x00,0x64,0x10,0x98]
        mulu    $2,$3,$4         # CHECK: mulu $2, $3, $4  # encoding: [0x00,0x64,0x10,0x99]
        neg       $2             # CHECK: neg  $2, $2            # encoding: [0x00,0x02,0x10,0x22]
        neg       $2, $3         # CHECK: neg  $2, $3            # encoding: [0x00,0x03,0x10,0x22]
        negu      $2             # CHECK: negu $2, $2            # encoding: [0x00,0x02,0x10,0x23]
        negu      $2,$3          # CHECK: negu $2, $3            # encoding: [0x00,0x03,0x10,0x23]
        not       $3, $4         # CHECK: not $3, $4             # encoding: [0x00,0x80,0x18,0x27]
        not       $3             # CHECK: not $3, $3             # encoding: [0x00,0x60,0x18,0x27]
        or      $2, 4            # CHECK: ori $2, $2, 4          # encoding: [0x34,0x42,0x00,0x04]
        pause                    # CHECK: pause                  # encoding:  [0x00,0x00,0x01,0x40]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} PAUSE>
        pref    1, 8($5)         # CHECK: pref 1, 8($5)          # encoding: [0x7c,0xa1,0x04,0x35]
        # FIXME: Use the code generator in order to print the .set directives
        #        instead of the instruction printer.
        rdhwr   $sp,$11          # CHECK:      .set  push
                                 # CHECK-NEXT: .set  mips32r2
                                 # CHECK-NEXT: rdhwr $sp, $11
                                 # CHECK-NEXT: .set  pop         # encoding: [0x7c,0x1d,0x58,0x3b]
        recip.d $f19,$f6         # CHECK: recip.d $f19, $f6      # encoding: [0x46,0x20,0x34,0xd5]
        recip.s $f3,$f30         # CHECK: recip.s $f3, $f30      # encoding: [0x46,0x00,0xf0,0xd5]
        rint.d $f2, $f4          # CHECK: rint.d $f2, $f4        # encoding: [0x46,0x20,0x20,0x9a]
        rint.s $f2, $f4          # CHECK: rint.s $f2, $f4        # encoding: [0x46,0x00,0x20,0x9a]
        round.w.d $f6, $f4       # CHECK: round.w.d $f6, $f4     # encoding: [0x46,0x20,0x21,0x8c]
                                 # CHECK:                        # <MCInst #{{.*}} ROUND_W_D64
        round.w.s $f27,$f28      # CHECK: round.w.s $f27, $f28   # encoding: [0x46,0x00,0xe6,0xcc]
                                 # CHECK:                        # <MCInst #{{.*}} ROUND_W_S
        rsqrt.s $f0,$f4          # CHECK: rsqrt.s $f0, $f4       # encoding: [0x46,0x00,0x20,0x16]
        rsqrt.d $f2,$f6          # CHECK: rsqrt.d $f2, $f6       # encoding: [0x46,0x20,0x30,0x96]
        s.s    $f2, 8($3)        # CHECK: swc1 $f2, 8($3)        # encoding: [0xe4,0x62,0x00,0x08]
        s.d    $f2, 8($3)        # CHECK: sdc1 $f2, 8($3)        # encoding: [0xf4,0x62,0x00,0x08]
        sc      $15,-40($s3)     # CHECK: sc $15, -40($19)       # encoding: [0x7e,0x6f,0xec,0x26]
        scd     $15,-51($sp)     # CHECK: scd $15, -51($sp)      # encoding: [0x7f,0xaf,0xe6,0xa7]
        sdbbp                    # CHECK: sdbbp                  # encoding: [0x00,0x00,0x00,0x0e]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SDBBP
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SDBBP_MM
        sdbbp     34             # CHECK: sdbbp 34               # encoding: [0x00,0x00,0x08,0x8e]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SDBBP
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SDBBP_MM
        sigrie                   # CHECK: sigrie                 # encoding: [0x04,0x17,0x00,0x00]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SIGRIE
        sigrie    257            # CHECK: sigrie 257             # encoding: [0x04,0x17,0x01,0x01]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SIGRIE
        sdc2    $20,629($s2)     # CHECK: sdc2 $20, 629($18)     # encoding: [0x49,0xf4,0x92,0x75]
        sel.d   $f0,$f1,$f2      # CHECK: sel.d $f0, $f1, $f2 # encoding: [0x46,0x22,0x08,0x10]
        sel.s   $f0,$f1,$f2      # CHECK: sel.s $f0, $f1, $f2 # encoding: [0x46,0x02,0x08,0x10]
        seleqz  $2,$3,$4         # CHECK: seleqz $2, $3, $4 # encoding: [0x00,0x64,0x10,0x35]
        seleqz.d $f0, $f2, $f4   # CHECK: seleqz.d $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x14]
        seleqz.s $f0, $f2, $f4   # CHECK: seleqz.s $f0, $f2, $f4 # encoding: [0x46,0x04,0x10,0x14]
        selnez  $2,$3,$4         # CHECK: selnez $2, $3, $4 # encoding: [0x00,0x64,0x10,0x37]
        selnez.d $f0, $f2, $f4   # CHECK: selnez.d $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x17]
        selnez.s $f0, $f2, $f4   # CHECK: selnez.s $f0, $f2, $f4 # encoding: [0x46,0x04,0x10,0x17]
        seb     $25, $15         # CHECK: seb $25, $15           # encoding: [0x7c,0x0f,0xcc,0x20]
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SEB_MM
        seb     $25              # CHECK: seb $25, $25           # encoding: [0x7c,0x19,0xcc,0x20]
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SEB_MM
        seh     $3, $12          # CHECK: seh $3, $12            # encoding: [0x7c,0x0c,0x1e,0x20]
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SEH_MM
        seh     $3               # CHECK: seh $3, $3             # encoding: [0x7c,0x03,0x1e,0x20]
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SEH_MM
        sgt     $4, $5           # CHECK: slt $4, $5, $4         # encoding: [0x00,0xa4,0x20,0x2a]
        sgt     $4, $5, $6       # CHECK: slt $4, $6, $5         # encoding: [0x00,0xc5,0x20,0x2a]
        sgtu    $4, $5           # CHECK: sltu $4, $5, $4        # encoding: [0x00,0xa4,0x20,0x2b]
        sgtu    $4, $5, $6       # CHECK: sltu $4, $6, $5        # encoding: [0x00,0xc5,0x20,0x2b]
        sll       $4, $5         # CHECK: sllv $4, $4, $5        # encoding: [0x00,0xa4,0x20,0x04]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SLL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SLL_MM
        sll       $a3,18         # CHECK: sll $7, $7, 18         # encoding: [0x00,0x07,0x3c,0x80]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SLL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SLL_MM
        sll       $a3,$zero,18   # CHECK: sll $7, $zero, 18      # encoding: [0x00,0x00,0x3c,0x80]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SLL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SLL_MM
        sll       $a3,$zero,$9   # CHECK: sllv $7, $zero, $9     # encoding: [0x01,0x20,0x38,0x04]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SLL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SLL_MM
        sllv      $a3,$zero,$9   # CHECK: sllv $7, $zero, $9     # encoding: [0x01,0x20,0x38,0x04]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SLLV
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SLLV_MM
        sra       $4, $5         # CHECK: srav $4, $4, $5        # encoding: [0x00,0xa4,0x20,0x07]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRA
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRA_MM
        sra       $s1,15         # CHECK: sra $17, $17, 15       # encoding: [0x00,0x11,0x8b,0xc3]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRA
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRA_MM
        sra       $s1,$s7,15     # CHECK: sra $17, $23, 15       # encoding: [0x00,0x17,0x8b,0xc3]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRA
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRA_MM
        sra       $s1,$s7,$sp    # CHECK: srav $17, $23, $sp     # encoding: [0x03,0xb7,0x88,0x07]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRA
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRA_MM
        srav      $s1,$s7,$sp    # CHECK: srav $17, $23, $sp     # encoding: [0x03,0xb7,0x88,0x07]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRAV
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRAV_MM
        srl       $4, $5         # CHECK: srlv $4, $4, $5        # encoding: [0x00,0xa4,0x20,0x06]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRL_MM
        srl       $2,7           # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRL_MM
        srl       $2,$2,7        # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRL_MM
        srl       $25,$s4,$a0    # CHECK: srlv $25, $20, $4      # encoding: [0x00,0x94,0xc8,0x06]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRL
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRL_MM
        srlv      $25,$s4,$a0    # CHECK: srlv $25, $20, $4      # encoding: [0x00,0x94,0xc8,0x06]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SRLV
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} SRLV_MM
        ssnop                    # CHECK: ssnop                  # encoding: [0x00,0x00,0x00,0x40]
        ssnop                    # WARNING: [[@LINE]]:9: warning: ssnop is deprecated for MIPS64r6 and is equivalent to a nop instruction
        swc2    $25,304($s0)     # CHECK: swc2 $25, 304($16)     # encoding: [0x49,0x79,0x81,0x30]
        sync                     # CHECK: sync                   # encoding: [0x00,0x00,0x00,0x0f]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SYNC
        sync    1                # CHECK: sync 1                 # encoding: [0x00,0x00,0x00,0x4f]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SYNC
        syscall                  # CHECK: syscall                # encoding: [0x00,0x00,0x00,0x0c]
        syscall   256            # CHECK: syscall 256            # encoding: [0x00,0x00,0x40,0x0c]
        tlbp                     # CHECK: tlbp                   # encoding: [0x42,0x00,0x00,0x08]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBP
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBP_MM
        tlbr                     # CHECK: tlbr                   # encoding: [0x42,0x00,0x00,0x01]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBR
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBR_MM
        tlbwi                    # CHECK: tlbwi                  # encoding: [0x42,0x00,0x00,0x02]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBWI
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBWI_MM
        tlbwr                    # CHECK: tlbwr                  # encoding: [0x42,0x00,0x00,0x06]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBWR
                                 # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBWR_MM
        teq     $0,$3            # CHECK: teq $zero, $3          # encoding: [0x00,0x03,0x00,0x34]
        teq     $5,$7,620        # CHECK: teq $5, $7, 620        # encoding: [0x00,0xa7,0x9b,0x34]
        tge     $5,$19,340       # CHECK: tge $5, $19, 340       # encoding: [0x00,0xb3,0x55,0x30]
        tge     $7,$10           # CHECK: tge $7, $10            # encoding: [0x00,0xea,0x00,0x30]
        tgeu    $20,$14,379      # CHECK: tgeu $20, $14, 379     # encoding: [0x02,0x8e,0x5e,0xf1]
        tgeu    $22,$28          # CHECK: tgeu $22, $gp          # encoding: [0x02,0xdc,0x00,0x31]
        tlt     $15,$13          # CHECK: tlt $15, $13           # encoding: [0x01,0xed,0x00,0x32]
        tlt     $2,$19,133       # CHECK: tlt $2, $19, 133       # encoding: [0x00,0x53,0x21,0x72]
        tltu    $11,$16          # CHECK: tltu $11, $16          # encoding: [0x01,0x70,0x00,0x33]
        tltu    $16,$29,1016     # CHECK: tltu $16, $sp, 1016    # encoding: [0x02,0x1d,0xfe,0x33]
        tne     $6,$17           # CHECK: tne $6, $17            # encoding: [0x00,0xd1,0x00,0x36]
        tne     $7,$8,885        # CHECK: tne $7, $8, 885        # encoding: [0x00,0xe8,0xdd,0x76]
        xor     $2, 4            # CHECK: xori $2, $2, 4         # encoding: [0x38,0x42,0x00,0x04]
        synci     -15842($a2)    # CHECK: synci -15842($6)       # encoding: [0x04,0xdf,0xc2,0x1e]
                                 # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} SYNCI
1:

        # Check that we accept traditional %relocation(symbol) offsets for stores
        # and loads, not just a sign 16 bit offset.

        lui     $2, %hi(g_8)            # CHECK:  encoding: [0x3c,0x02,A,A]
        lb      $3, %lo(g_8)($2)        # CHECK:  encoding: [0x80,0x43,A,A]
        lh      $3, %lo(g_8)($2)        # CHECK:  encoding: [0x84,0x43,A,A]
        lhu     $3, %lo(g_8)($2)        # CHECK:  encoding: [0x94,0x43,A,A]
        lw      $3, %lo(g_8)($2)        # CHECK:  encoding: [0x8c,0x43,A,A]
        sb      $3, %lo(g_8)($2)        # CHECK:  encoding: [0xa0,0x43,A,A]
        sh      $3, %lo(g_8)($2)        # CHECK:  encoding: [0xa4,0x43,A,A]
        sw      $3, %lo(g_8)($2)        # CHECK:  encoding: [0xac,0x43,A,A]

        lwc1    $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xc4,0x40,A,A]
        ldc1    $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xd4,0x40,A,A]
        swc1    $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xe4,0x40,A,A]
        sdc1    $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xf4,0x40,A,A]
        lwu     $3, %lo(g_8)($2)        # CHECK:  encoding: [0x9c,0x43,A,A]
        ld      $3, %lo(g_8)($2)        # CHECK:  encoding: [0xdc,0x43,A,A]
        sd      $3, %lo(g_8)($2)        # CHECK:  encoding: [0xfc,0x43,A,A]
        .type   g_8,@object
        .comm   g_8,16,16
