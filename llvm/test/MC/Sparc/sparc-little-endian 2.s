! RUN: llvm-mc %s -triple=sparcel-linux-gnu -show-encoding | FileCheck %s
! RUN: llvm-mc -triple=sparcel-linux-gnu -filetype=obj < %s | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-OBJ

        ! CHECK-OBJ: .text:
        .BB0:

        ! Ensure instructions are emitted in reversed byte order:

        ! CHECK: call %g1     ! encoding: [0x00,0x40,0xc0,0x9f]
        ! CHECK-OBJ: 0: 00 40 c0 9f  call %g1
        call %g1

        ! ...and that fixups are applied to the correct bytes.

        ! CHECK: ba .BB0      ! encoding: [A,A,0b10AAAAAA,0x10]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        ! CHECK-OBJ: 4: ff ff bf 10  ba 4194303
        ba .BB0
