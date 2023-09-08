! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: membar #LoadLoad | #StoreLoad | #LoadStore | #StoreStore  ! encoding: [0x81,0x43,0xe0,0x0f]
        membar 15

        ! CHECK: membar #LoadLoad  ! encoding: [0x81,0x43,0xe0,0x01]
        membar #LoadLoad

        ! CHECK: membar #LoadLoad | #StoreStore  ! encoding: [0x81,0x43,0xe0,0x09]
        membar #LoadLoad | #StoreStore

        ! CHECK: membar #LoadLoad | #StoreLoad | #LoadStore | #StoreStore | #Lookaside | #MemIssue | #Sync  ! encoding: [0x81,0x43,0xe0,0x7f]
        membar #LoadLoad | #StoreLoad | #LoadStore | #StoreStore | #Lookaside | #MemIssue | #Sync

        ! CHECK: swapa [%i0+6] %asi, %o2   ! encoding: [0xd4,0xfe,0x20,0x06]
        swapa [%i0+6] %asi, %o2

        ! CHECK: ldstuba [%i0+2] %asi, %g1 ! encoding: [0xc2,0xee,0x20,0x02]
        ldstuba [%i0+2] %asi, %g1
