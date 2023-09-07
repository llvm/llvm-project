! RUN: not llvm-mc %s -arch=sparc   -show-encoding 2>&1 | FileCheck %s --check-prefix=V8
! RUN: not llvm-mc %s -arch=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefix=V9

! V8: error: malformed ASI tag, must be a constant integer expression
! V8-NEXT: lduba [%i0] asi, %o2
! V9: error: malformed ASI tag, must be %asi or a constant integer expression
! V9-NEXT: lduba [%i0] asi, %o2
lduba [%i0] asi, %o2

! V8: error: malformed ASI tag, must be a constant integer expression
! V8-NEXT: lduba [%i0] %g0, %o2
! V9: error: malformed ASI tag, must be %asi or a constant integer expression
! V9-NEXT: lduba [%i0] %g0, %o2
lduba [%i0] %g0, %o2

! V8: error: malformed ASI tag, must be a constant integer expression
! V8-NEXT: lduba [%i0] %0, %o2
! V9: error: malformed ASI tag, must be %asi or a constant integer expression
! V9-NEXT: lduba [%i0] %0, %o2
lduba [%i0] %0, %o2

! V8: error: invalid ASI number, must be between 0 and 255
! V8-NEXT: lduba [%i0] -1, %o2
! V9: error: invalid ASI number, must be between 0 and 255
! V9-NEXT: lduba [%i0] -1, %o2
lduba [%i0] -1, %o2

! V8: error: invalid ASI number, must be between 0 and 255
! V8-NEXT: lduba [%i0] 256, %o2
! V9: error: invalid ASI number, must be between 0 and 255
! V9-NEXT: lduba [%i0] 256, %o2
lduba [%i0] 256, %o2

!! %asi register is only introduced in V9
! V8: error: malformed ASI tag, must be a constant integer expression
! V8-NEXT: lduba [%i0] %asi, %o2
lduba [%i0] %asi, %o2

!! [Reg+Imm] can't be used with immediate ASI forms.
! V8: error: invalid operand for instruction
! V8-NEXT: lduba [%i0+1] 255, %o2
! V9: error: invalid operand for instruction
! V9-NEXT: lduba [%i0+1] 255, %o2
lduba [%i0+1] 255, %o2

!! [Reg+Reg] can't be used with stored tag in %asi.
! V9: error: invalid operand for instruction
! V9-NEXT: lduba [%i0+%i1] %asi, %o2
lduba [%i0+%i1] %asi, %o2
