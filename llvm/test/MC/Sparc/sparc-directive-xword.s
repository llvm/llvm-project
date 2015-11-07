! RUN: not llvm-mc %s -arch=sparc   -show-encoding 2>&1 | FileCheck %s --check-prefix=SPARC32
! RUN: llvm-mc %s -triple sparc64 -show-encoding | FileCheck %s --check-prefix=SPARC64
! RUN: llvm-mc %s -triple sparcv9 -show-encoding | FileCheck %s --check-prefix=SPARCV9

        ! SPARC32:       error: unknown directive
        ! SPARC32-NEXT:  .xword 65536
        ! SPARC32-NEXT:  ^

        ! SPARC64:  .xword 65536
        .xword 65536

        ! SPARCV9:  .xword 65536
        .xword 65536
