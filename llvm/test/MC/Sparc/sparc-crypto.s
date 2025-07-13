! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefixes=NO-CRYPTO --implicit-check-not=error:
! RUN: llvm-mc %s -triple=sparcv9 -mattr=+crypto -show-encoding | FileCheck %s --check-prefixes=CRYPTO

!! Crypto instructions.

! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_eround01 %f0, %f2, %f4, %f6         ! encoding: [0x8c,0xc8,0x08,0x02]
aes_eround01 %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_eround23 %f0, %f2, %f4, %f6         ! encoding: [0x8c,0xc8,0x08,0x22]
aes_eround23 %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_dround01 %f0, %f2, %f4, %f6         ! encoding: [0x8c,0xc8,0x08,0x42]
aes_dround01 %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_dround23 %f0, %f2, %f4, %f6         ! encoding: [0x8c,0xc8,0x08,0x62]
aes_dround23 %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_eround01_l %f0, %f2, %f4, %f6    ! encoding: [0x8c,0xc8,0x08,0x82]
aes_eround01_l %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_eround23_l %f0, %f2, %f4, %f6    ! encoding: [0x8c,0xc8,0x08,0xa2]
aes_eround23_l %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_dround01_l %f0, %f2, %f4, %f6    ! encoding: [0x8c,0xc8,0x08,0xc2]
aes_dround01_l %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_dround23_l %f0, %f2, %f4, %f6    ! encoding: [0x8c,0xc8,0x08,0xe2]
aes_dround23_l %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_kexpand0 %f0, %f2, %f4              ! encoding: [0x89,0xb0,0x26,0x02]
aes_kexpand0 %f0, %f2, %f4
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_kexpand1 %f0, %f2, 4, %f6           ! encoding: [0x8c,0xc8,0x09,0x02]
aes_kexpand1 %f0, %f2, 4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: aes_kexpand2 %f0, %f2, %f4              ! encoding: [0x89,0xb0,0x26,0x22]
aes_kexpand2 %f0, %f2, %f4

! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: camellia_f %f0, %f2, %f4, %f6           ! encoding: [0x8c,0xc8,0x09,0x82]
camellia_f %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: camellia_fl %f0, %f2, %f4               ! encoding: [0x89,0xb0,0x27,0x82]
camellia_fl %f0, %f2, %f4
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: camellia_fli %f0, %f2, %f4              ! encoding: [0x89,0xb0,0x27,0xa2]
camellia_fli %f0, %f2, %f4

! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: crc32c %f0, %f2, %f4                    ! encoding: [0x89,0xb0,0x28,0xe2]
crc32c %f0, %f2, %f4

! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: des_round %f0, %f2, %f4, %f6            ! encoding: [0x8c,0xc8,0x09,0x22]
des_round %f0, %f2, %f4, %f6
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: des_ip %f0, %f2                         ! encoding: [0x85,0xb0,0x26,0x80]
des_ip %f0, %f2
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: des_iip %f0, %f2                        ! encoding: [0x85,0xb0,0x26,0xa0]
des_iip %f0, %f2
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: des_kexpand %f0, 2, %f4                 ! encoding: [0x89,0xb0,0x26,0xc2]
des_kexpand %f0, 2, %f4

! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: md5                                     ! encoding: [0x81,0xb0,0x28,0x00]
md5
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: sha1                                    ! encoding: [0x81,0xb0,0x28,0x20]
sha1
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: sha256                                  ! encoding: [0x81,0xb0,0x28,0x40]
sha256
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: sha512                                  ! encoding: [0x81,0xb0,0x28,0x60]
sha512

! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: mpmul 1                                 ! encoding: [0x81,0xb0,0x29,0x01]
mpmul 1
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: montmul 2                               ! encoding: [0x81,0xb0,0x29,0x22]
montmul 2
! NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
! CRYPTO: montsqr 3                               ! encoding: [0x81,0xb0,0x29,0x43]
montsqr 3
