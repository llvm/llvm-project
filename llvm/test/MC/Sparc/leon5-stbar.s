! RUN: llvm-mc %s -triple=sparc -mcpu=leon5 -show-encoding \
! RUN:   | FileCheck %s --check-prefixes=CHECK,ENC
! RUN: llvm-mc %s -triple=sparc -mcpu=leon5 -filetype=obj \
! RUN:   | llvm-objdump -d --mcpu=leon5 - | FileCheck %s

! The LEON5 STBAR takes barrier options and the write-combining hint for the
! following store in its otherwise unused bits 12:0. Without options it is the
! plain STBAR.

! CHECK: stbar
! ENC-SAME: encoding: [0x81,0x43,0xc0,0x00]
stbar

! CHECK: stbar #WriteCombine
! ENC-SAME: encoding: [0x81,0x43,0xc1,0x00]
stbar #WriteCombine

! CHECK: stbar #StoreLoad | #Sync
! ENC-SAME: encoding: [0x81,0x43,0xc0,0x42]
stbar #StoreLoad | #Sync

! CHECK: stbar #Lookaside | #MemIssue | #WriteCombine
! ENC-SAME: encoding: [0x81,0x43,0xc1,0x30]
stbar #Lookaside|#MemIssue|#WriteCombine

! A mask number is printed with the tags.
! CHECK: stbar #StoreLoad | #Lookaside | #MemIssue | #Sync | #WriteCombine
! ENC-SAME: encoding: [0x81,0x43,0xc1,0x72]
stbar 0x172
