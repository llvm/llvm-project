! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9

!! Short names
! V9: casxa [%i0] #ASI_N, %l6, %o2            ! encoding: [0xd5,0xf6,0x00,0x96]
casxa [%i0] #ASI_N, %l6, %o2
! V9: casxa [%i0] #ASI_N_L, %l6, %o2          ! encoding: [0xd5,0xf6,0x01,0x96]
casxa [%i0] #ASI_N_L, %l6, %o2
! V9: casxa [%i0] #ASI_AIUP, %l6, %o2         ! encoding: [0xd5,0xf6,0x02,0x16]
casxa [%i0] #ASI_AIUP, %l6, %o2
! V9: casxa [%i0] #ASI_AIUS, %l6, %o2         ! encoding: [0xd5,0xf6,0x02,0x36]
casxa [%i0] #ASI_AIUS, %l6, %o2
! V9: casxa [%i0] #ASI_AIUP_L, %l6, %o2       ! encoding: [0xd5,0xf6,0x03,0x16]
casxa [%i0] #ASI_AIUP_L, %l6, %o2
! V9: casxa [%i0] #ASI_AIUS_L, %l6, %o2       ! encoding: [0xd5,0xf6,0x03,0x36]
casxa [%i0] #ASI_AIUS_L, %l6, %o2
! V9: casx [%i0], %l6, %o2                    ! encoding: [0xd5,0xf6,0x10,0x16]
casxa [%i0] #ASI_P, %l6, %o2
! V9: casxa [%i0] #ASI_S, %l6, %o2            ! encoding: [0xd5,0xf6,0x10,0x36]
casxa [%i0] #ASI_S, %l6, %o2
! V9: casxa [%i0] #ASI_PNF, %l6, %o2          ! encoding: [0xd5,0xf6,0x10,0x56]
casxa [%i0] #ASI_PNF, %l6, %o2
! V9: casxa [%i0] #ASI_SNF, %l6, %o2          ! encoding: [0xd5,0xf6,0x10,0x76]
casxa [%i0] #ASI_SNF, %l6, %o2
! V9: casxl [%i0], %l6, %o2                   ! encoding: [0xd5,0xf6,0x11,0x16]
casxa [%i0] #ASI_P_L, %l6, %o2
! V9: casxa [%i0] #ASI_S_L, %l6, %o2          ! encoding: [0xd5,0xf6,0x11,0x36]
casxa [%i0] #ASI_S_L, %l6, %o2
! V9: casxa [%i0] #ASI_PNF_L, %l6, %o2        ! encoding: [0xd5,0xf6,0x11,0x56]
casxa [%i0] #ASI_PNF_L, %l6, %o2
! V9: casxa [%i0] #ASI_SNF_L, %l6, %o2        ! encoding: [0xd5,0xf6,0x11,0x76]
casxa [%i0] #ASI_SNF_L, %l6, %o2

!! Long names
! V9: casxa [%i0] #ASI_N, %l6, %o2            ! encoding: [0xd5,0xf6,0x00,0x96]
casxa [%i0] #ASI_NUCLEUS, %l6, %o2
! V9: casxa [%i0] #ASI_N_L, %l6, %o2          ! encoding: [0xd5,0xf6,0x01,0x96]
casxa [%i0] #ASI_NUCLEUS_LITTLE, %l6, %o2
! V9: casxa [%i0] #ASI_AIUP, %l6, %o2         ! encoding: [0xd5,0xf6,0x02,0x16]
casxa [%i0] #ASI_AS_IF_USER_PRIMARY, %l6, %o2
! V9: casxa [%i0] #ASI_AIUS, %l6, %o2         ! encoding: [0xd5,0xf6,0x02,0x36]
casxa [%i0] #ASI_AS_IF_USER_SECONDARY, %l6, %o2
! V9: casxa [%i0] #ASI_AIUP_L, %l6, %o2       ! encoding: [0xd5,0xf6,0x03,0x16]
casxa [%i0] #ASI_AS_IF_USER_PRIMARY_LITTLE, %l6, %o2
! V9: casxa [%i0] #ASI_AIUS_L, %l6, %o2       ! encoding: [0xd5,0xf6,0x03,0x36]
casxa [%i0] #ASI_AS_IF_USER_SECONDARY_LITTLE, %l6, %o2
! V9: casx [%i0], %l6, %o2                    ! encoding: [0xd5,0xf6,0x10,0x16]
casxa [%i0] #ASI_PRIMARY, %l6, %o2
! V9: casxa [%i0] #ASI_S, %l6, %o2            ! encoding: [0xd5,0xf6,0x10,0x36]
casxa [%i0] #ASI_SECONDARY, %l6, %o2
! V9: casxa [%i0] #ASI_PNF, %l6, %o2          ! encoding: [0xd5,0xf6,0x10,0x56]
casxa [%i0] #ASI_PRIMARY_NOFAULT, %l6, %o2
! V9: casxa [%i0] #ASI_SNF, %l6, %o2          ! encoding: [0xd5,0xf6,0x10,0x76]
casxa [%i0] #ASI_SECONDARY_NOFAULT, %l6, %o2
! V9: casxl [%i0], %l6, %o2                   ! encoding: [0xd5,0xf6,0x11,0x16]
casxa [%i0] #ASI_PRIMARY_LITTLE, %l6, %o2
! V9: casxa [%i0] #ASI_S_L, %l6, %o2          ! encoding: [0xd5,0xf6,0x11,0x36]
casxa [%i0] #ASI_SECONDARY_LITTLE, %l6, %o2
! V9: casxa [%i0] #ASI_PNF_L, %l6, %o2        ! encoding: [0xd5,0xf6,0x11,0x56]
casxa [%i0] #ASI_PRIMARY_NOFAULT_LITTLE, %l6, %o2
! V9: casxa [%i0] #ASI_SNF_L, %l6, %o2        ! encoding: [0xd5,0xf6,0x11,0x76]
casxa [%i0] #ASI_SECONDARY_NOFAULT_LITTLE, %l6, %o2
