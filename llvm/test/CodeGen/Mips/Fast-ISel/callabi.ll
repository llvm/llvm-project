; RUN: llc -mtriple=mipsel -mcpu=mips32 -O0 -relocation-model=pic \
; RUN:     -fast-isel-abort=3 -verify-machineinstrs < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,32R1
; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -O0 -relocation-model=pic \
; RUN:     -fast-isel-abort=3 -verify-machineinstrs < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,32R2

declare void @xb(i8)

define void @cxb() {
  ; ALL-LABEL:    cxb:

  ; ALL:            addiu   $[[T0:[0-9]+]], $zero, 10

  ; 32R1:           sll     $[[T1:[0-9]+]], $[[T0]], 24
  ; 32R1:           sra     $4, $[[T1]], 24

  ; 32R2:           seb     $4, $[[T0]]
  call void @xb(i8 10)
  ret void
}

declare void @xh(i16)

define void @cxh() {
  ; ALL-LABEL:    cxh:

  ; ALL:            addiu   $[[T0:[0-9]+]], $zero, 10

  ; 32R1:           sll     $[[T1:[0-9]+]], $[[T0]], 16
  ; 32R1:           sra     $4, $[[T1]], 16

  ; 32R2:           seh     $4, $[[T0]]
  call void @xh(i16 10)
  ret void
}

declare void @xi(i32)

define void @cxi() {
  ; ALL-LABEL:    cxi:

  ; ALL-DAG:        addiu   $4, $zero, 10
  ; ALL-DAG:        lw      $25, %got(xi)(${{[0-9]+}})
  ; ALL:            jalr    $25
  call void @xi(i32 10)
  ret void
}

declare void @xbb(i8, i8)

define void @cxbb() {
  ; ALL-LABEL:    cxbb:

  ; ALL-DAG:        addiu   $[[T0:[0-9]+]], $zero, 76
  ; ALL-DAG:        addiu   $[[T1:[0-9]+]], $zero, 101

  ; 32R1-DAG:       sll     $[[T2:[0-9]+]], $[[T0]], 24
  ; 32R1-DAG:       sra     $[[T3:[0-9]+]], $[[T2]], 24
  ; 32R1-DAG:       sll     $[[T4:[0-9]+]], $[[T1]], 24
  ; 32R1-DAG:       sra     $[[T5:[0-9]+]], $[[T4]], 24

  ; 32R2-DAG:       seb     $4, $[[T0]]
  ; 32R2-DAG:       seb     $5, $[[T1]]
  call void @xbb(i8 76, i8 101)
  ret void
}

declare void @xhh(i16, i16)

define void @cxhh() {
  ; ALL-LABEL:    cxhh:

  ; ALL-DAG:        addiu   $[[T0:[0-9]+]], $zero, 76
  ; ALL-DAG:        addiu   $[[T1:[0-9]+]], $zero, 101

  ; 32R1-DAG:       sll     $[[T2:[0-9]+]], $[[T0]], 16
  ; 32R1-DAG:       sra     $[[T3:[0-9]+]], $[[T2]], 16
  ; 32R1-DAG:       sll     $[[T4:[0-9]+]], $[[T1]], 16
  ; 32R1-DAG:       sra     $[[T5:[0-9]+]], $[[T4]], 16

  ; 32R2-DAG:       seh     $4, $[[T0]]
  ; 32R2-DAG:       seh     $5, $[[T1]]
  call void @xhh(i16 76, i16 101)
  ret void
}

declare void @xii(i32, i32)

define void @cxii() {
  ; ALL-LABEL:    cxii:

  ; ALL-DAG:        addiu   $4, $zero, 746
  ; ALL-DAG:        addiu   $5, $zero, 892
  ; ALL-DAG:        lw      $25, %got(xii)(${{[0-9]+}})
  ; ALL:            jalr    $25
  call void @xii(i32 746, i32 892)
  ret void
}

declare void @xccc(i8, i8, i8)

define void @cxccc() {
  ; ALL-LABEL:    cxccc:

  ; ALL-DAG:        addiu   $[[T0:[0-9]+]], $zero, 88
  ; ALL-DAG:        addiu   $[[T1:[0-9]+]], $zero, 44
  ; ALL-DAG:        addiu   $[[T2:[0-9]+]], $zero, 11

  ; 32R1-DAG:       sll     $[[T3:[0-9]+]], $[[T0]], 24
  ; 32R1-DAG:       sra     $4, $[[T3]], 24
  ; 32R1-DAG:       sll     $[[T4:[0-9]+]], $[[T1]], 24
  ; 32R1-DAG:       sra     $5, $[[T4]], 24
  ; 32R1-DAG:       sll     $[[T5:[0-9]+]], $[[T2]], 24
  ; 32R1-DAG:       sra     $6, $[[T5]], 24

  ; 32R2-DAG:       seb     $4, $[[T0]]
  ; 32R2-DAG:       seb     $5, $[[T1]]
  ; 32R2-DAG:       seb     $6, $[[T2]]
  call void @xccc(i8 88, i8 44, i8 11)
  ret void
}

declare void @xhhh(i16, i16, i16)

define void @cxhhh() {
  ; ALL-LABEL:    cxhhh:

  ; ALL-DAG:        addiu   $[[T0:[0-9]+]], $zero, 88
  ; ALL-DAG:        addiu   $[[T1:[0-9]+]], $zero, 44
  ; ALL-DAG:        addiu   $[[T2:[0-9]+]], $zero, 11

  ; 32R1-DAG:       sll     $[[T3:[0-9]+]], $[[T0]], 16
  ; 32R1-DAG:       sra     $4, $[[T3]], 16
  ; 32R1-DAG:       sll     $[[T4:[0-9]+]], $[[T1]], 16
  ; 32R1-DAG:       sra     $5, $[[T4]], 16
  ; 32R1-DAG:       sll     $[[T5:[0-9]+]], $[[T2]], 16
  ; 32R1-DAG:       sra     $6, $[[T5]], 16

  ; 32R2-DAG:       seh     $4, $[[T0]]
  ; 32R2-DAG:       seh     $5, $[[T1]]
  ; 32R2-DAG:       seh     $6, $[[T2]]
  call void @xhhh(i16 88, i16 44, i16 11)
  ret void
}

declare void @xiii(i32, i32, i32)

define void @cxiii() {
  ; ALL-LABEL:    cxiii:

  ; ALL-DAG:        addiu   $4, $zero, 88
  ; ALL-DAG:        addiu   $5, $zero, 44
  ; ALL-DAG:        addiu   $6, $zero, 11
  ; ALL-DAG:        lw      $25, %got(xiii)(${{[0-9]+}})
  ; ALL:            jalr    $25
  call void @xiii(i32 88, i32 44, i32 11)
  ret void
}

declare void @xcccc(i8, i8, i8, i8)

define void @cxcccc() {
  ; ALL-LABEL:    cxcccc:

  ; ALL:        addiu   $[[R1:[0-9]+]], $zero, 88
  ; ALL:        addiu   $[[R2:[0-9]+]], $zero, 44
  ; ALL:        addiu   $[[R3:[0-9]+]], $zero, 11
  ; ALL:        addiu   $[[R4:[0-9]+]], $zero, 33
  ; 32R1:       sll     $[[R1:[0-9]+]], $[[R1]], 24
  ; 32R1:       sra     $4, $[[R1]], 24
  ; 32R2:       seb     $4, $[[R1]]
  ; 32R1:       sll     $[[R2]], $[[R2]], 24
  ; 32R1:       sra     $5, $[[R2]], 24
  ; 32R2:       seb     $5, $[[R2]]
  ; 32R1:       sll     $[[R3]], $[[R3]], 24
  ; 32R1:       sra     $6, $[[R3]], 24
  ; 32R2:       seb     $6, $[[R3]]
  ; 32R1:       sll     $[[R4]], $[[R4]], 24
  ; 32R1:       sra     $7, $[[R4]], 24
  ; 32R2:       seb     $7, $[[R4]]

  ; ALL:        lw      $25, %got(xcccc)(${{[0-9]+}})
  ; ALL:        jalr    $25
  ; ALL:        jr      $ra
  call void @xcccc(i8 88, i8 44, i8 11, i8 33)
  ret void
}

declare void @xhhhh(i16, i16, i16, i16)

define void @cxhhhh() {
  ; ALL-LABEL:    cxhhhh:

  ; ALL:        addiu   $[[R1:[0-9]+]], $zero, 88
  ; ALL:        addiu   $[[R2:[0-9]+]], $zero, 44
  ; ALL:        addiu   $[[R3:[0-9]+]], $zero, 11
  ; ALL:        addiu   $[[R4:[0-9]+]], $zero, 33
  ; 32R1:       sll     $[[R1]], $[[R1]], 16
  ; 32R1:       sra     $4, $[[R1]], 16
  ; 32R2:       seh     $4, $[[R1]]
  ; 32R1:       sll     $[[R2]], $[[R2]], 16
  ; 32R1:       sra     $5, $[[R2]], 16
  ; 32R2:       seh     $5, $[[R2]]
  ; 32R1:       sll     $[[R3]], $[[R3]], 16
  ; 32R1:       sra     $6, $[[R3]], 16
  ; 32R2:       seh     $6, $[[R3]]
  ; 32R1:       sll     $[[R4]], $[[R4]], 16
  ; 32R1:       sra     $7, $[[R4]], 16
  ; 32R2:       seh     $7, $[[R4]]

  ; ALL:        lw      $25, %got(xhhhh)(${{[0-9]+}})
  ; ALL:        jalr    $25
  ; ALL:        jr      $ra

  call void @xhhhh(i16 88, i16 44, i16 11, i16 33)
  ret void
}

declare void @xiiii(i32, i32, i32, i32)

define void @cxiiii() {
  ; ALL-LABEL:    cxiiii:

  ; ALL-DAG:        addiu   $4, $zero, 167
  ; ALL-DAG:        addiu   $5, $zero, 320
  ; ALL-DAG:        addiu   $6, $zero, 97
  ; ALL-DAG:        addiu   $7, $zero, 14
  ; ALL-DAG:        lw      $25, %got(xiiii)(${{[0-9]+}})
  ; ALL:            jalr    $25
  call void @xiiii(i32 167, i32 320, i32 97, i32 14)
  ret void
}

@c1 = global i8 -45, align 1
@uc1 = global i8 27, align 1
@s1 = global i16 -1789, align 2
@us1 = global i16 1256, align 2

define void @cxiiiiconv() {
  ; ALL-LABEL:    cxiiiiconv:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lw      $[[REG_C1_ADDR:[0-9]+]], %got(c1)($[[REG_GP]])
  ; ALL-DAG:        lbu     $[[REG_C1:[0-9]+]], 0($[[REG_C1_ADDR]])
  ; 32R1-DAG:       sll     $[[REG_C1_1:[0-9]+]], $[[REG_C1]], 24
  ; 32R1-DAG:       sra     $4, $[[REG_C1_1]], 24
  ; 32R2-DAG:       seb     $4, $[[REG_C1]]
  ; FIXME: andi is superfulous
  ; ALL-DAG:        lw      $[[REG_UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
  ; ALL-DAG:        lbu     $[[REG_UC1:[0-9]+]], 0($[[REG_UC1_ADDR]])
  ; ALL-DAG:        andi    $5, $[[REG_UC1]], 255
  ; ALL-DAG:        lw      $[[REG_S1_ADDR:[0-9]+]], %got(s1)($[[REG_GP]])
  ; ALL-DAG:        lhu     $[[REG_S1:[0-9]+]], 0($[[REG_S1_ADDR]])
  ; 32R1-DAG:       sll     $[[REG_S1_1:[0-9]+]], $[[REG_S1]], 16
  ; 32R1-DAG:       sra     $6, $[[REG_S1_1]], 16
  ; 32R2-DAG:       seh     $6, $[[REG_S1]]
  ; FIXME andi is superfulous
  ; ALL-DAG:        lw      $[[REG_US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
  ; ALL-DAG:        lhu     $[[REG_US1:[0-9]+]], 0($[[REG_US1_ADDR]])
  ; ALL-DAG:        andi    $7, $[[REG_US1]], 65535
  ; ALL:            jalr    $25
  %1 = load i8, i8* @c1, align 1
  %conv = sext i8 %1 to i32
  %2 = load i8, i8* @uc1, align 1
  %conv1 = zext i8 %2 to i32
  %3 = load i16, i16* @s1, align 2
  %conv2 = sext i16 %3 to i32
  %4 = load i16, i16* @us1, align 2
  %conv3 = zext i16 %4 to i32
  call void @xiiii(i32 %conv, i32 %conv1, i32 %conv2, i32 %conv3)
  ret void
}

declare void @xf(float)

define void @cxf() {
  ; ALL-LABEL:    cxf:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL:            lui     $[[REG_FPCONST_1:[0-9]+]], 17886
  ; ALL:            ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 17067
  ; ALL:            mtc1    $[[REG_FPCONST]], $f12
  ; ALL:            lw      $25, %got(xf)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xf(float 0x40BBC85560000000)
  ret void
}

declare void @xff(float, float)

define void @cxff() {
  ; ALL-LABEL:    cxff:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lui     $[[REG_FPCONST_1:[0-9]+]], 16314
  ; ALL-DAG:        ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 21349
  ; ALL-DAG:        mtc1    $[[REG_FPCONST]], $f12
  ; ALL-DAG:        lui     $[[REG_FPCONST_2:[0-9]+]], 16593
  ; ALL-DAG:        ori     $[[REG_FPCONST_3:[0-9]+]], $[[REG_FPCONST_2]], 24642
  ; ALL-DAG:        mtc1    $[[REG_FPCONST_3]], $f14
  ; ALL-DAG:        lw      $25, %got(xff)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xff(float 0x3FF74A6CA0000000, float 0x401A2C0840000000)
  ret void
}

declare void @xfi(float, i32)

define void @cxfi() {
  ; ALL-LABEL:    cxfi:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lui     $[[REG_FPCONST_1:[0-9]+]], 16540
  ; ALL-DAG:        ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 33554
  ; ALL-DAG:        mtc1    $[[REG_FPCONST]], $f12
  ; ALL-DAG:        addiu   $5, $zero, 102
  ; ALL-DAG:        lw      $25, %got(xfi)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xfi(float 0x4013906240000000, i32 102)
  ret void
}

declare void @xfii(float, i32, i32)

define void @cxfii() {
  ; ALL-LABEL:    cxfii:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lui     $[[REG_FPCONST_1:[0-9]+]], 17142
  ; ALL-DAG:        ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 16240
  ; ALL-DAG:        mtc1    $[[REG_FPCONST]], $f12
  ; ALL-DAG:        addiu   $5, $zero, 9993
  ; ALL-DAG:        addiu   $6, $zero, 10922
  ; ALL-DAG:        lw      $25, %got(xfii)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xfii(float 0x405EC7EE00000000, i32 9993, i32 10922)
  ret void
}

declare void @xfiii(float, i32, i32, i32)

define void @cxfiii() {
  ; ALL-LABEL:    cxfiii:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lui     $[[REG_FPCONST_1:[0-9]+]], 17120
  ; ALL-DAG:        ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 14681
  ; ALL-DAG:        mtc1    $[[REG_FPCONST]], $f12
  ; ALL-DAG:        addiu   $5, $zero, 3948
  ; ALL-DAG:        lui     $[[REG_I_1:[0-9]+]], 1
  ; ALL-DAG:        ori     $6, $[[REG_I_1]], 23475
  ; ALL-DAG:        lui     $[[REG_I_2:[0-9]+]], 1
  ; ALL-DAG:        ori     $7, $[[REG_I_2]], 45686
  ; ALL-DAG:        lw      $25, %got(xfiii)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xfiii(float 0x405C072B20000000, i32 3948, i32 89011, i32 111222)
  ret void
}

declare void @xd(double)

define void @cxd() {
  ; ALL-LABEL:    cxd:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lui     $[[REG_FPCONST_1:[0-9]+]], 16514
  ; ALL-DAG:        ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 48037
  ; ALL-DAG:        lui     $[[REG_FPCONST_3:[0-9]+]], 58195
  ; ALL-DAG:        ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 63439
  ; ALL-DAG:        mtc1    $[[REG_FPCONST_4]], $f12
  ; 32R1-DAG:       mtc1    $[[REG_FPCONST_2]], $f13
  ; 32R2-DAG:       mthc1   $[[REG_FPCONST_2]], $f12
  ; ALL-DAG:        lw      $25, %got(xd)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xd(double 5.994560e+02)
  ret void
}

declare void @xdd(double, double)

define void @cxdd() {
  ; ALL-LABEL:    cxdd:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lui     $[[REG_FPCONST_1:[0-9]+]], 16531
  ; ALL-DAG:        ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 19435
  ; ALL-DAG:        lui     $[[REG_FPCONST_3:[0-9]+]], 34078
  ; ALL-DAG:        ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 47186
  ; ALL-DAG:        mtc1    $[[REG_FPCONST_4]], $f12
  ; 32R1-DAG:       mtc1    $[[REG_FPCONST_2]], $f13
  ; 32R2-DAG:       mthc1   $[[REG_FPCONST_2]], $f12
  ; ALL-DAG:        lui     $[[REG_FPCONST_1:[0-9]+]], 16629
  ; ALL-DAG:        ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 45873
  ; ALL-DAG:        lui     $[[REG_FPCONST_3:[0-9]+]], 63438
  ; ALL-DAG:        ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 55575
  ; ALL-DAG:        mtc1    $[[REG_FPCONST_4]], $f14
  ; 32R1-DAG:       mtc1    $[[REG_FPCONST_2]], $f15
  ; 32R2-DAG:       mthc1   $[[REG_FPCONST_2]], $f14
  ; ALL-DAG:        lw      $25, %got(xdd)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xdd(double 1.234980e+03, double 0x40F5B331F7CED917)
  ret void
}

declare void @xif(i32, float)

define void @cxif() {
  ; ALL-LABEL:    cxif:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        addiu   $4, $zero, 345
  ; ALL-DAG:        lui     $[[REGF_1:[0-9]+]], 17374
  ; ALL-DAG:        ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 29393
  ; ALL-DAG:        mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
  ; ALL-DAG:        mfc1    $5, $f[[REGF_3]]
  ; ALL-DAG:        lw      $25, %got(xif)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xif(i32 345, float 0x407BCE5A20000000)
  ret void
}

declare void @xiff(i32, float, float)

define void @cxiff() {
  ; ALL-LABEL:    cxiff:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        addiu   $4, $zero, 12239
  ; ALL-DAG:        lui     $[[REGF0_1:[0-9]+]], 17526
  ; ALL-DAG:        ori     $[[REGF0_2:[0-9]+]], $[[REGF0_1]], 55706
  ; ALL-DAG:        mtc1    $[[REGF0_2]], $f[[REGF0_3:[0-9]+]]
  ; ALL-DAG:        lui     $[[REGF1_1:[0-9]+]], 16543
  ; ALL-DAG:        ori     $[[REGF1_2:[0-9]+]], $[[REGF1_1]], 65326
  ; ALL:            mtc1    $[[REGF1_2]], $f[[REGF1_3:[0-9]+]]
  ; ALL-DAG:        mfc1    $5, $f[[REGF0_3]]
  ; ALL-DAG:        mfc1    $6, $f[[REGF1_3]]
  ; ALL-DAG:        lw      $25, %got(xiff)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xiff(i32 12239, float 0x408EDB3340000000, float 0x4013FFE5C0000000)
  ret void
}

declare void @xifi(i32, float, i32)

define void @cxifi() {
  ; ALL-LABEL:    cxifi:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        addiu   $4, $zero, 887
  ; ALL-DAG:        lui     $[[REGF_1:[0-9]+]], 16659
  ; ALL-DAG:        ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 48759
  ; ALL-DAG:        mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
  ; ALL-DAG:        mfc1    $5, $f[[REGF_3]]
  ; ALL-DAG:        addiu   $6, $zero, 888
  ; ALL-DAG:        lw      $25, %got(xifi)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xifi(i32 887, float 0x402277CEE0000000, i32 888)
  ret void
}

declare void @xifif(i32, float, i32, float)

define void @cxifif() {
  ; ALL-LABEL:    cxifif:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        lui     $[[REGI:[0-9]+]], 1
  ; ALL-DAG:        ori     $4, $[[REGI]], 2238
  ; ALL-DAG:        lui     $[[REGF0_1:[0-9]+]], 17527
  ; ALL-DAG:        ori     $[[REGF0_2:[0-9]+]], $[[REGF0_1]], 2015
  ; ALL-DAG:        mtc1    $[[REGF0_2]], $f[[REGF0_3:[0-9]+]]
  ; ALL-DAG:        addiu   $6, $zero, 9991
  ; ALL-DAG:        lui     $[[REGF1_1:[0-9]+]], 17802
  ; ALL-DAG:        ori     $[[REGF1_2:[0-9]+]], $[[REGF1_1]], 58470
  ; ALL:            mtc1    $[[REGF1_2]], $f[[REGF1_3:[0-9]+]]
  ; ALL-DAG:        mfc1    $5, $f[[REGF0_3]]
  ; ALL-DAG:        mfc1    $7, $f[[REGF1_3]]
  ; ALL-DAG:        lw      $25, %got(xifif)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xifif(i32 67774, float 0x408EE0FBE0000000,
                   i32 9991, float 0x40B15C8CC0000000)
  ret void
}

declare void @xiffi(i32, float, float, i32)

define void @cxiffi() {
  ; ALL-LABEL:    cxiffi:

  ; ALL:            addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:        addiu   $4, $zero, 45
  ; ALL-DAG:        lui     $[[REGF0_1:[0-9]+]], 16307
  ; ALL-DAG:        ori     $[[REGF0_2:[0-9]+]], $[[REGF0_1]], 13107
  ; ALL-DAG:        mtc1    $[[REGF0_2]], $f[[REGF0_3:[0-9]+]]
  ; ALL-DAG:        lui     $[[REGF1_1:[0-9]+]], 17529
  ; ALL-DAG:        ori     $[[REGF1_2:[0-9]+]], $[[REGF1_1]], 39322
  ; ALL:            mtc1    $[[REGF1_2]], $f[[REGF1_3:[0-9]+]]
  ; ALL-DAG:        mfc1    $5, $f[[REGF0_3]]
  ; ALL-DAG:        mfc1    $6, $f[[REGF1_3]]
  ; ALL-DAG:        addiu   $7, $zero, 234
  ; ALL-DAG:        lw      $25, %got(xiffi)($[[REG_GP]])
  ; ALL:            jalr    $25
  call void @xiffi(i32 45, float 0x3FF6666660000000,
                   float 0x408F333340000000, i32 234)
  ret void
}

declare void @xifii(i32, float, i32, i32)

define void @cxifii() {
  ; ALL-LABEL:    cxifii:

  ; ALL-DAG:    addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
  ; ALL-DAG:    addiu   $4, $zero, 12239
  ; ALL-DAG:    lui     $[[REGF_1:[0-9]+]], 17526
  ; ALL-DAG:    ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 55706
  ; ALL-DAG:    mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
  ; ALL-DAG:    mfc1    $5, $f[[REGF_3]]
  ; ALL-DAG:    lui     $[[REGI2:[0-9]+]], 15
  ; ALL-DAG:    ori     $6, $[[REGI2]], 15837
  ; ALL-DAG:    addiu   $7, $zero, 1234
  ; ALL-DAG:    lw      $25, %got(xifii)($[[REG_GP]])
  ; ALL:        jalr    $25
  call void @xifii(i32 12239, float 0x408EDB3340000000, i32 998877, i32 1234)
  ret void
}
