! RUN: not llvm-mc %s -triple=sparc -show-encoding 2>&1 | FileCheck %s --check-prefix=V8
! RUN: llvm-mc %s -triple=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx 1, %g1, %o1
! V9: mov 1, %o1                     ! encoding: [0x92,0x10,0x20,0x01]
setx 1, %g1, %o1

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx (0+1), %g1, %o1
! V9: mov 1, %o1                     ! encoding: [0x92,0x10,0x20,0x01]
setx (0+1), %g1, %o1

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx -1, %g1, %o1
! V9: mov -1, %o1                     ! encoding: [0x92,0x10,0x3f,0xff]
setx -1, %g1, %o1

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx (0-1), %g1, %o1
! V9: mov -1, %o1                     ! encoding: [0x92,0x10,0x3f,0xff]
setx (0-1), %g1, %o1

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx 0xffffffff, %g1, %o1
! V9: sethi %hi(4294967295), %o1      ! encoding: [0x13,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hi(4294967295), kind: fixup_sparc_hi22
! V9: or %o1, %lo(4294967295), %o1    ! encoding: [0x92,0x12,0b011000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %lo(4294967295), kind: fixup_sparc_lo10
setx 0xffffffff, %g1, %o1

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx (0xffff0000+0x0000ffff), %g1, %o1
! V9: sethi %hi(4294967295), %o1      ! encoding: [0x13,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hi(4294967295), kind: fixup_sparc_hi22
! V9: or %o1, %lo(4294967295), %o1    ! encoding: [0x92,0x12,0b011000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %lo(4294967295), kind: fixup_sparc_lo10
setx (0xffff0000+0x0000ffff), %g1, %o1

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx 0x0123456789abcdef, %g1, %o0
! V9: sethi %hi(81985529216486895), %o0       ! encoding: [0x11,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hi(81985529216486895), kind: fixup_sparc_hi22
! V9: or %o0, %lo(81985529216486895), %o0     ! encoding: [0x90,0x12,0b001000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %lo(81985529216486895), kind: fixup_sparc_lo10
! V9: sethi %hh(81985529216486895), %g1       ! encoding: [0x03,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hh(81985529216486895), kind: fixup_sparc_hh
! V9: or %g1, %hm(81985529216486895), %g1     ! encoding: [0x82,0x10,0b011000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %hm(81985529216486895), kind: fixup_sparc_hm
! V9: sllx %g1, 32, %g1                       ! encoding: [0x83,0x28,0x70,0x20]
! V9: or %g1, %o0, %o0                        ! encoding: [0x90,0x10,0x40,0x08]
setx 0x0123456789abcdef, %g1, %o0

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx (0x0123456700000000+0x0000000089abcdef), %g1, %o0
! V9: sethi %hi(81985529216486895), %o0       ! encoding: [0x11,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hi(81985529216486895), kind: fixup_sparc_hi22
! V9: or %o0, %lo(81985529216486895), %o0     ! encoding: [0x90,0x12,0b001000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %lo(81985529216486895), kind: fixup_sparc_lo10
! V9: sethi %hh(81985529216486895), %g1       ! encoding: [0x03,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hh(81985529216486895), kind: fixup_sparc_hh
! V9: or %g1, %hm(81985529216486895), %g1     ! encoding: [0x82,0x10,0b011000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %hm(81985529216486895), kind: fixup_sparc_hm
! V9: sllx %g1, 32, %g1                       ! encoding: [0x83,0x28,0x70,0x20]
! V9: or %g1, %o0, %o0                        ! encoding: [0x90,0x10,0x40,0x08]
setx (0x0123456700000000+0x0000000089abcdef), %g1, %o0

! V8:      error: instruction requires a CPU feature not currently enabled
! V8-NEXT: setx (.BB1-.BB0), %g1, %o0
! V9: sethi %hi(.BB1-.BB0), %o0               ! encoding: [0x11,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hi(.BB1-.BB0), kind: fixup_sparc_hi22
! V9: or %o0, %lo(.BB1-.BB0), %o0             ! encoding: [0x90,0x12,0b001000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %lo(.BB1-.BB0), kind: fixup_sparc_lo10
! V9: sethi %hh(.BB1-.BB0), %g1               ! encoding: [0x03,0b00AAAAAA,A,A]
! V9:                                 !   fixup A - offset: 0, value: %hh(.BB1-.BB0), kind: fixup_sparc_hh
! V9: or %g1, %hm(.BB1-.BB0), %g1             ! encoding: [0x82,0x10,0b011000AA,A]
! V9:                                 !   fixup A - offset: 0, value: %hm(.BB1-.BB0), kind: fixup_sparc_hm
! V9: sllx %g1, 32, %g1                       ! encoding: [0x83,0x28,0x70,0x20]
! V9: or %g1, %o0, %o0                        ! encoding: [0x90,0x10,0x40,0x08]
setx (.BB1-.BB0), %g1, %o0
